#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benign PCAP Processor - 每个文件处理完后刷新所有会话

主要特点：
- 处理完每个PCAP文件后，立即刷新所有活跃会话到MongoDB
- 会话不会跨文件合并
- 存储格式与原版完全相同

用法:
    python benign_pcap_processor_flush_per_file.py --pcap-dir ./benign_pcaps/ --label benign
"""
import os
import gc
import logging
import argparse
from collections import defaultdict

import pymongo
from scapy.all import PcapReader, IP, TCP, UDP, ARP, ICMP

# ======================== CICIDS2018 STRONG-DRIFT FILTER ========================

# Enable via CLI: --cicids-2018
CICIDS_2018_MODE = True

# Known malicious/public IPs for your strong-drift days:
# 2018-02-14 Bruteforce: attackers 18.221.219.4, 13.58.98.64; victim 172.31.69.25
# 2018-03-02 Botnet Ares: master/C2 18.219.211.138 (plus optional slaves)
CICIDS_2018_BAD_IPS = {
    "18.221.219.4",
    "13.58.98.64",
    "172.31.69.25",
    "18.219.211.138",
    "18.222.10.237",
    "18.222.86.193",
    "18.222.62.221",
    "13.59.9.106",
    "18.222.102.2",
    "18.219.212.0",
    "18.216.105.13",
    "18.219.163.126",
    "18.216.164.12",
}

def is_bad_cicids2018_key(key):
    """Return True if this 5-tuple key touches known CICIDS2018 attack IPs."""
    if not CICIDS_2018_MODE:
        return False
    ip1 = key[0]
    ip2 = key[2]
    return (ip1 in CICIDS_2018_BAD_IPS) or (ip2 in CICIDS_2018_BAD_IPS)

MAX_PACKETS_TO_STORE = 1000

# SESSION PARAMETERS
MIN_PACKET_COUNT = 2  # Filter out single-packet sessions
MAX_PACKET_COUNT = 10000
TIMEOUT = 30  # Session timeout in seconds

# QUALITY FILTERS
MIN_SESSION_DURATION = 0  # 10ms minimum (filter out instant sessions)
MIN_TOTAL_PAYLOAD = 10  # Minimum total payload bytes
REQUIRE_BIDIRECTIONAL = False  # Set True to require packets in both directions

# PROTOCOL FILTERS (what to include)
ALLOWED_PROTOCOLS = {'tcp', 'udp'}  # Only TCP/UDP
EXCLUDE_BROADCAST = True
EXCLUDE_MULTICAST = True

# SPECIFIC PROTOCOL FILTERING
FILTER_COMMON_NOISE = True  # Filter common noise protocols
NOISE_PORTS = {
    67, 68,  # DHCP
    137, 138,  # NetBIOS
    5353,  # mDNS
    1900,  # SSDP
    # 53,        # DNS (uncomment to exclude)
}

# PERFORMANCE
GC_INTERVAL = 100000
BATCH_SIZE = 100
LOG_INTERVAL = 100000

# MONGODB
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "IDS_2018"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger("BenignProcessor")


# ======================== UTILS ========================
def bytes2str(info):
    """Convert bytes to hex string."""
    if isinstance(info, bytes):
        return info.hex()
    return bytes(info).hex()


def get_payload_bytes(packet):
    """Extract payload bytes."""
    if packet.haslayer("Raw"):
        return list(packet["Raw"].load)
    return []


def mask_and_extract(pkt):
    """Mask addresses and extract header/payload."""
    header_bytes = []

    # Mask and collect
    if pkt.haslayer("Ether"):
        eth = pkt["Ether"]
        eth.src = "00:00:00:00:00:00"
        eth.dst = "00:00:00:00:00:00"
        header_bytes.extend(bytes(eth))

    if pkt.haslayer("IP"):
        ip = pkt["IP"]
        ip.src = "0.0.0.0"
        ip.dst = "0.0.0.0"
        header_bytes.extend(bytes(ip))

    # Protocol detection
    proto = 'Other'
    if pkt.haslayer("TCP"):
        header_bytes.extend(bytes(pkt["TCP"]))
        proto = 'TCP'
    elif pkt.haslayer("UDP"):
        header_bytes.extend(bytes(pkt["UDP"]))
        proto = 'UDP'
    elif pkt.haslayer("ICMP"):
        proto = 'ICMP'

    payload_bytes = get_payload_bytes(pkt)

    return header_bytes, payload_bytes, proto


def process_packet_fast(pkt):
    """Process packet - minimal operations."""
    header_bytes, payload_bytes, proto = mask_and_extract(pkt)

    total_length = len(bytes(pkt))
    header_length = len(header_bytes)
    payload_length = len(payload_bytes)  # 1. Keep the TRUE length for statistics

    # ===================== FIX START =====================
    # 2. Truncate only the bytes we intend to save to the database
    # limit to 2048 bytes (2KB)
    if len(payload_bytes) > 2048:
        payload_bytes_for_storage = payload_bytes[:2048]
    else:
        payload_bytes_for_storage = payload_bytes
    # ===================== FIX END =======================

    p_header = bytes2str(header_bytes)
    p_payload = bytes2str(payload_bytes_for_storage) # 3. Convert the TRUNCATED bytes to hex

    return p_header, p_payload, total_length, header_length, payload_length, proto


# ======================== FILTERING ========================

def is_broadcast_multicast(pkt):
    """Check if packet is broadcast or multicast."""
    if not pkt.haslayer(IP):
        return True

    ip = pkt[IP]
    dst_ip = ip.dst

    # Broadcast
    if dst_ip.endswith('.255'):
        return True

    # Multicast (224.0.0.0 to 239.255.255.255)
    first_octet = int(dst_ip.split('.')[0])
    if 224 <= first_octet <= 239:
        return True

    return False


def is_noise_traffic(pkt):
    """Check if packet is noise traffic (DHCP, NetBIOS, etc.)."""
    if not FILTER_COMMON_NOISE:
        return False

    if pkt.haslayer(TCP):
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
    elif pkt.haslayer(UDP):
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
    else:
        return False

    # Check if either port is in noise ports
    if sport in NOISE_PORTS or dport in NOISE_PORTS:
        return True

    return False


def should_skip_packet(pkt):
    """
    Determine if packet should be filtered out.
    Returns: True if should skip, False if should process
    """
    # Must have IP layer
    if not pkt.haslayer(IP):
        return True

    # Check protocol
    if pkt.haslayer(TCP):
        proto = 'tcp'
    elif pkt.haslayer(UDP):
        proto = 'udp'
    elif pkt.haslayer(ICMP):
        proto = 'icmp'
    else:
        return True  # Unknown protocol

    # Protocol filter
    if proto not in ALLOWED_PROTOCOLS:
        return True

    # Broadcast/Multicast filter
    if EXCLUDE_BROADCAST and is_broadcast_multicast(pkt):
        return True

    # Noise traffic filter
    if is_noise_traffic(pkt):
        return True

    return False


def is_quality_session(sess):
    """
    Check if session meets quality criteria.
    Returns: (True/False, reason)
    """
    pkt_count = len(sess["p_header"])
    duration = sess["last_ts"] - sess["first_ts"]
    total_payload = sum(sess["p_pp"])

    # Minimum packet count
    if pkt_count < MIN_PACKET_COUNT:
        return False, "too_few_packets"

    # Maximum packet count
    if pkt_count > MAX_PACKET_COUNT:
        return False, "too_many_packets"

    # Minimum duration
    if duration < MIN_SESSION_DURATION:
        return False, "too_short"

    # Minimum payload
    if total_payload < MIN_TOTAL_PAYLOAD:
        return False, "no_payload"

    # Bidirectional check
    if REQUIRE_BIDIRECTIONAL:
        # Check if we have packets in both directions
        # For TCP: check if we have different flag patterns
        # For UDP: approximate by checking packet size variation
        protocols = sess["protocols"]
        packet_sizes = sess["p_tt"]

        if 'TCP' in protocols:
            # Simple heuristic: if all packets are same size, likely unidirectional
            if len(set(packet_sizes)) < 2:
                return False, "unidirectional"
        elif 'UDP' in protocols:
            # UDP bidirectional check: need size variation or packet count > 5
            if len(set(packet_sizes)) < 2 and pkt_count < 5:
                return False, "unidirectional"

    return True, None


# ======================== SESSION MANAGEMENT ========================

def process_benign_pcaps(file_paths, label, collection):
    """
    Process benign PCAP files with quality filtering.
    **MODIFIED**: Flushes all sessions when switching to next file.
    """
    batch = []
    total_sessions = 0
    quality_sessions = 0
    filtered_sessions = 0
    filter_reasons = defaultdict(int)
    packets_processed = 0
    packets_filtered = 0

    # Closure for flush
    def flush_session(key, sess):
        nonlocal batch, total_sessions, quality_sessions, filtered_sessions

        # 1. Capture Raw Statistics
        raw_pkt_count = len(sess["p_header"])
        total_sessions += 1

        # 2. Quality check (Perform on FULL session data)
        is_quality, reason = is_quality_session(sess)

        if not is_quality:
            filtered_sessions += 1
            filter_reasons[reason] += 1
            return

        if is_bad_cicids2018_key(key):
            filtered_sessions += 1
            filter_reasons["cicids2018_attack_ip"] += 1
            return

        quality_sessions += 1

        # ==================== TRUNCATION LOGIC START ====================
        # If packets > MAX_PACKETS_TO_STORE, we slice the lists.
        # Otherwise, we take the whole list.

        limit = MAX_PACKETS_TO_STORE

        # We must slice ALL parallel arrays to keep index alignment
        final_p_header = sess["p_header"][:limit]
        final_p_payload = sess["p_payload"][:limit]
        final_p_tt = sess["p_tt"][:limit]  # Total Lengths
        final_p_hh = sess["p_hh"][:limit]  # Header Lengths
        final_p_pp = sess["p_pp"][:limit]  # Payload Lengths
        final_protocols = sess["protocols"][:limit]

        final_count = len(final_p_header)

        # RECALCULATE Payload Length
        # This is crucial: The stored 'payload_length' must match the sum of stored packets
        final_payload_sum = sum(final_p_pp)
        # ==================== TRUNCATION LOGIC END ======================

        doc = {
            "key": {
                "ip1": key[0], "port1": key[1],
                "ip2": key[2], "port2": key[3],
                "proto": key[4]
            },
            "first_ts": sess["first_ts"],
            "last_ts": sess["last_ts"],  # We keep original last_ts to show true flow duration
            "duration": sess["last_ts"] - sess["first_ts"],

            "packet_count": final_count,  # e.g., 1000
            "original_packet_count": raw_pkt_count,  # e.g., 5321 (Useful for analytics)

            "p_header": final_p_header,
            "p_payload": final_p_payload,
            "p_tt": final_p_tt,
            "p_hh": final_p_hh,
            "p_pp": final_p_pp,
            "protocols": final_protocols,
            "payload_length": final_payload_sum,  # Recalculated sum
            "label": label
        }

        if CICIDS_2018_MODE:
            doc["is_CICIDS_2018"] = True

        batch.append(doc)

        if len(batch) >= BATCH_SIZE:
            try:
                collection.insert_many(batch, ordered=False)
            except Exception as e:
                logger.error(f"Insert error: {e}")
            batch.clear()

    # ======================== MAIN PROCESSING LOOP ========================
    logger.info("Starting PCAP processing (flush-per-file mode)...")

    for file_idx, file_path in enumerate(file_paths, 1):
        filename = os.path.basename(file_path)
        logger.info(f"Processing file {file_idx}/{len(file_paths)}: {filename}")

        # *** KEY CHANGE: Initialize fresh session dict for EACH file ***
        active_sessions = {}
        file_packets = 0

        try:
            with PcapReader(file_path) as pcap:
                for pkt in pcap:
                    packets_processed += 1
                    file_packets += 1

                    # Logging
                    if packets_processed % LOG_INTERVAL == 0:
                        logger.info(f"  Processed {packets_processed:,} packets, "
                                    f"{len(active_sessions)} active sessions, "
                                    f"{quality_sessions:,} quality sessions saved")
                        gc.collect()

                    # Filter packets
                    if should_skip_packet(pkt):
                        packets_filtered += 1
                        continue

                    # Get 5-tuple key
                    if not pkt.haslayer(IP):
                        continue

                    ip = pkt[IP]
                    src = ip.src
                    dst = ip.dst

                    if pkt.haslayer(TCP):
                        sport = pkt[TCP].sport
                        dport = pkt[TCP].dport
                        proto = 'TCP'
                    elif pkt.haslayer(UDP):
                        sport = pkt[UDP].sport
                        dport = pkt[UDP].dport
                        proto = 'UDP'
                    else:
                        continue

                    ts = float(pkt.time)

                    # Normalize 5-tuple
                    if (src, sport) < (dst, dport):
                        key = (src, sport, dst, dport, proto)
                    else:
                        key = (dst, dport, src, sport, proto)

                    # Check session timeout
                    sess = active_sessions.get(key)
                    if sess is not None:
                        if (ts - sess["last_ts"]) > TIMEOUT:
                            flush_session(key, sess)
                            sess = None

                    if sess is None:
                        # New session
                        sess = {
                            "first_ts": ts,
                            "last_ts": ts,
                            "p_header": [],
                            "p_payload": [],
                            "p_tt": [],
                            "p_hh": [],
                            "p_pp": [],
                            "protocols": [],
                        }
                        active_sessions[key] = sess

                    # Process packet
                    try:
                        p_header, p_payload, p_tt, p_hh, p_pp, proto = process_packet_fast(pkt)
                    except:
                        continue

                    # Update session
                    sess["last_ts"] = ts
                    sess["p_header"].append(p_header)
                    sess["p_payload"].append(p_payload)
                    sess["p_tt"].append(p_tt)
                    sess["p_hh"].append(p_hh)
                    sess["p_pp"].append(p_pp)
                    sess["protocols"].append(proto)

        except Exception as e:
            logger.error(f"Error in {filename}: {e}")

        # *** KEY CHANGE: Flush ALL sessions from this file before moving to next ***
        logger.info(f"  File complete. Flushing {len(active_sessions)} sessions from {filename}...")
        for key, sess in list(active_sessions.items()):
            flush_session(key, sess)

        # Clear the dictionary for next file
        active_sessions.clear()

        logger.info(f"  ✓ {filename}: {file_packets:,} packets processed")

    # Insert final batch
    if batch:
        try:
            collection.insert_many(batch, ordered=False)
        except Exception as e:
            logger.error(f"Final insert error: {e}")

    # Print filter statistics
    logger.info("=" * 60)
    logger.info("FILTERING STATISTICS")
    logger.info(f"  Packets processed: {packets_processed:,}")
    logger.info(f"  Packets filtered: {packets_filtered:,} ({packets_filtered / packets_processed * 100:.1f}%)")
    logger.info(f"  Sessions created: {total_sessions:,}")
    logger.info(f"  Sessions filtered: {filtered_sessions:,}")
    logger.info(f"  Quality sessions: {quality_sessions:,}")

    if filter_reasons:
        logger.info("\n  Filter reasons:")
        for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1]):
            logger.info(f"    - {reason}: {count:,}")

    logger.info("=" * 60)


def create_indexes(collection):
    """Create indexes - same as attack processor."""
    try:
        collection.create_index([("label", 1)], background=True)
        collection.create_index([("first_ts", 1)], background=True)
        collection.create_index([("last_ts", 1)], background=True)
        collection.create_index([("duration", 1)], background=True)
        collection.create_index([("packet_count", 1)], background=True)
        collection.create_index([
            ("key.ip1", 1), ("key.port1", 1),
            ("key.ip2", 1), ("key.port2", 1),
            ("key.proto", 1)
        ], background=True)
        logger.info("✓ Indexes created")
    except Exception as e:
        logger.warning(f"Index error: {e}")


# ======================== MAIN ========================

def parse_args():
    parser = argparse.ArgumentParser(description="High-quality benign PCAP processor (flush per file)")
    parser.add_argument("--pcap-dir", required=True, help="Directory containing benign PCAP files")
    parser.add_argument("--label", default="benign", help="Label for sessions (default: benign)")
    parser.add_argument("--mongo-uri", default=MONGO_URI, help="MongoDB URI")
    parser.add_argument("--db-name", default=DB_NAME, help="Database name")
    parser.add_argument("--collection", default="benign", help="Collection name")

    # Quality filters
    parser.add_argument("--min-packets", type=int, default=MIN_PACKET_COUNT,
                        help=f"Minimum packets per session (default: {MIN_PACKET_COUNT})")
    parser.add_argument("--min-payload", type=int, default=MIN_TOTAL_PAYLOAD,
                        help=f"Minimum total payload bytes (default: {MIN_TOTAL_PAYLOAD})")
    parser.add_argument("--min-duration", type=float, default=MIN_SESSION_DURATION,
                        help=f"Minimum session duration in seconds (default: {MIN_SESSION_DURATION})")
    parser.add_argument("--timeout", type=int, default=TIMEOUT,
                        help=f"Session timeout in seconds (default: {TIMEOUT})")

    # Filters
    parser.add_argument("--include-dns", action="store_true", help="Include DNS traffic")
    parser.add_argument("--include-icmp", action="store_true", help="Include ICMP traffic")
    parser.add_argument("--no-filter-noise", action="store_true", help="Don't filter noise protocols")
    parser.add_argument("--allow-broadcast", action="store_true", help="Allow broadcast/multicast")

    parser.add_argument("--skip-indexes", action="store_true", help="Skip index creation")

    parser.add_argument(
        "--cicids-2018",
        action="store_true",
        help="Tag docs with is_CICIDS_2018=True and filter known CICIDS2018 attack IPs"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Update global settings based on arguments
    global MIN_PACKET_COUNT, MIN_TOTAL_PAYLOAD, MIN_SESSION_DURATION, TIMEOUT
    global FILTER_COMMON_NOISE, EXCLUDE_BROADCAST, ALLOWED_PROTOCOLS, NOISE_PORTS
    global CICIDS_2018_MODE

    CICIDS_2018_MODE = args.cicids_2018
    logger.info(f"CICIDS2018 mode: {CICIDS_2018_MODE}")

    MIN_PACKET_COUNT = args.min_packets
    MIN_TOTAL_PAYLOAD = args.min_payload
    MIN_SESSION_DURATION = args.min_duration
    TIMEOUT = args.timeout
    FILTER_COMMON_NOISE = not args.no_filter_noise
    EXCLUDE_BROADCAST = not args.allow_broadcast

    if args.include_icmp:
        ALLOWED_PROTOCOLS.add('icmp')

    if args.include_dns:
        NOISE_PORTS.discard(53)

    logger.info("=" * 60)
    logger.info("HIGH-QUALITY BENIGN PCAP PROCESSOR (FLUSH PER FILE)")
    logger.info(f"Label: {args.label}")
    logger.info(f"Directory: {args.pcap_dir}")
    logger.info(f"Min packets: {MIN_PACKET_COUNT}")
    logger.info(f"Min payload: {MIN_TOTAL_PAYLOAD}B")
    logger.info(f"Min duration: {MIN_SESSION_DURATION}s")
    logger.info(f"Timeout: {TIMEOUT}s")
    logger.info(f"Protocols: {ALLOWED_PROTOCOLS}")
    logger.info("=" * 60)

    # Get PCAP files
    pcap_files = [
        os.path.join(args.pcap_dir, f)
        for f in os.listdir(args.pcap_dir)
    ]

    if not pcap_files:
        logger.error("No .pcap files found")
        return

    pcap_files.sort()
    logger.info(f"Found {len(pcap_files)} PCAP files")

    # Connect to MongoDB
    client = None
    try:
        logger.info("Connecting to MongoDB...")
        client = pymongo.MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        logger.info("✓ Connected")

        db = client[args.db_name]
        collection = db[args.collection]

        # Check existing data
        existing = collection.count_documents({})
        if existing > 0:
            logger.warning(f"Collection has {existing} documents")
            response = input("Drop collection? (y/n): ")
            if response.lower() == 'y':
                db.drop_collection(args.collection)
                collection = db[args.collection]
                logger.info("✓ Collection dropped")

        # Process benign PCAPs
        process_benign_pcaps(
            file_paths=pcap_files,
            label=args.label,
            collection=collection
        )

        # Create indexes
        if not args.skip_indexes:
            create_indexes(collection)

        # Final statistics
        final_count = collection.count_documents({})

        logger.info("=" * 60)
        logger.info("✓ SUCCESS")
        logger.info(f"  Collection: {args.db_name}.{args.collection}")
        logger.info(f"  Total sessions saved: {final_count:,}")
        logger.info(f"  Label: {args.label}")
        logger.info("=" * 60)

        # Sample statistics
        if final_count > 0:
            pipeline = [
                {"$group": {
                    "_id": None,
                    "avg_packets": {"$avg": "$packet_count"},
                    "avg_payload": {"$avg": "$payload_length"},
                    "avg_duration": {"$avg": "$duration"},
                }}
            ]
            stats = list(collection.aggregate(pipeline))
            if stats:
                logger.info("Session Statistics:")
                logger.info(f"  Avg packets: {stats[0]['avg_packets']:.1f}")
                logger.info(f"  Avg payload: {stats[0]['avg_payload']:.1f} bytes")
                logger.info(f"  Avg duration: {stats[0]['avg_duration']:.3f} seconds")
                logger.info("=" * 60)

    except pymongo.errors.ServerSelectionTimeoutError:
        logger.error("❌ MongoDB connection failed")
    except KeyboardInterrupt:
        logger.warning("⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            client.close()


if __name__ == '__main__':
    main()