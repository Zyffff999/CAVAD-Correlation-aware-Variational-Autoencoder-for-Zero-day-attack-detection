#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST PCAP Directory Processor - Maximum speed, minimal overhead
Process all PCAP files in a directory quickly without timestamp analysis.

OPTIMIZATIONS:
- No timestamp sorting (uses filename order)
- Reduced logging
- Larger batch sizes
- Less frequent garbage collection
- Optimized packet processing

USAGE:
    python fast_pcap_processor.py --attack-name backdoor --csv data.csv --pcap-dir ./pcaps/
"""

import os
import gc
import logging
import argparse
from collections import defaultdict

import pymongo
import pandas as pd
from tqdm import tqdm
from scapy.all import PcapReader, IP, TCP, UDP

# ======================== OPTIMIZED CONFIG ========================

MIN_PACKET_COUNT = 1
MAX_PACKET_COUNT = 1000

# OPTIMIZED for speed
GC_INTERVAL = 100000  # Increased from 50K to 100K
BATCH_SIZE = 200  # Increased from 300 to 500
LOG_INTERVAL = 100000  # Log every 100K packets

TIMEOUT = 5
TIME_BUFFER = 5

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "NOT_IOT-1"

# Minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger("FastProcessor")


# ======================== OPTIMIZED UTILS ========================

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
    """
    Combined function: mask addresses + extract header/payload in one pass.
    OPTIMIZATION: Reduces function calls and layer access.
    """
    header_bytes = []

    # Mask and collect in one pass
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

    # Protocol detection and header extraction
    proto = 'Other'
    if pkt.haslayer("TCP"):
        header_bytes.extend(bytes(pkt["TCP"]))
        proto = 'TCP'
    elif pkt.haslayer("UDP"):
        header_bytes.extend(bytes(pkt["UDP"]))
        proto = 'UDP'
    elif pkt.haslayer("ICMP"):
        proto = 'ICMP'

    # Payload
    payload_bytes = get_payload_bytes(pkt)

    return header_bytes, payload_bytes, proto


def process_packet_fast(pkt):
    """
    OPTIMIZED packet processing - minimal operations.
    """
    header_bytes, payload_bytes, proto = mask_and_extract(pkt)

    total_length = len(bytes(pkt))
    header_length = len(header_bytes)
    payload_length = len(payload_bytes)

    p_header = bytes2str(header_bytes)
    p_payload = bytes2str(payload_bytes)

    return p_header, p_payload, total_length, header_length, payload_length, proto


# ======================== 5-TUPLE ========================

def normalize_proto_csv(v):
    """Normalize proto from CSV."""
    s = str(v).strip().lower()
    if s in ('tcp', '6', '6.0', 'tcp(6)'):
        return 'tcp'
    if s in ('udp', '17', '17.0', 'udp(17)'):
        return 'udp'
    if s in ('icmp', '1', '1.0'):
        return 'icmp'
    return s


def get_5tuple_key(pkt):
    """Extract 5-tuple key."""
    if not pkt.haslayer(IP):
        return None

    ip = pkt[IP]
    src_ip = ip.src
    dst_ip = ip.dst

    if pkt.haslayer(TCP):
        l4 = pkt[TCP]
        proto_norm = 'tcp'
    elif pkt.haslayer(UDP):
        l4 = pkt[UDP]
        proto_norm = 'udp'
    else:
        return None

    sport = int(l4.sport)
    dport = int(l4.dport)

    # Normalize direction


    a = (src_ip, sport)
    b = (dst_ip, dport)
    if a <= b:
        return (src_ip, sport, dst_ip, dport, proto_norm)
    else:
        return (dst_ip, dport, src_ip, sport, proto_norm)


# ======================== CSV CONSTRAINTS ========================

def normalize_attack_name(name):
    return str(name).strip().lower()


def load_constraints(csv_path, attack_name):
    """Load constraints from CSV."""
    attack_name_norm = normalize_attack_name(attack_name)
    df = pd.read_csv(csv_path)

    df["type_norm"] = df["type"].astype(str).str.strip().str.lower()
    sub = df[df["type_norm"] == attack_name_norm]

    if sub.empty:
        logger.warning(f"No constraints for '{attack_name_norm}'")
        return {}

    constraints = {}

    for _, row in sub.iterrows():
        proto_norm = normalize_proto_csv(row["proto"])
        ip1 = str(row["ip1"])
        ip2 = str(row["ip2"])
        port1 = int(row["port1"])
        port2 = int(row["port2"])

        a = (ip1, port1)
        b = (ip2, port2)
        if a <= b:
            key = (ip1, port1, ip2, port2, proto_norm)
        else:
            key = (ip2, port2, ip1, port1, proto_norm)

        t_min = float(row["min_ts"]) - TIME_BUFFER
        t_max = float(row["max_ts"]) + TIME_BUFFER

        if key not in constraints:
            constraints[key] = [t_min, t_max]
        else:
            constraints[key][0] = min(constraints[key][0], t_min)
            constraints[key][1] = max(constraints[key][1], t_max)

    constraints = {k: (v[0], v[1]) for k, v in constraints.items()}
    logger.info(f"Loaded {len(constraints)} constraints for '{attack_name_norm}'")
    return constraints


def label_session_online(sess, key, ts, constraints, attack_name_norm):
    """Label session if matches constraints."""
    if sess["label"] == attack_name_norm:
        return

    if key not in constraints:
        return

    t_min, t_max = constraints[key]
    if t_min <= ts <= t_max:
        sess["label"] = attack_name_norm


# ======================== FAST PROCESSOR ========================

def process_directory_fast(file_paths, attack_name, constraints, collection, store_full_bytes=False):
    """
    OPTIMIZED: Fast processing with minimal overhead.
    """

    attack_name_norm = normalize_attack_name(attack_name)
    batch = []

    # Statistics
    total_sessions = 0
    attack_sessions = 0
    benign_sessions = 0
    packets_processed = 0

    # Sessions persist across files
    active_sessions = {}

    def flush_session(key, sess):
        """Flush session to batch."""
        nonlocal batch, total_sessions, attack_sessions, benign_sessions

        pkt_count = len(sess["p_tt"])
        if not (MIN_PACKET_COUNT <= pkt_count <= MAX_PACKET_COUNT):
            return

        # Truncate if needed
        if pkt_count > 1000:
            sess["p_header"] = sess["p_header"][:1000]
            sess["p_payload"] = sess["p_payload"][:1000]
            sess["p_tt"] = sess["p_tt"][:1000]
            sess["p_hh"] = sess["p_hh"][:1000]
            sess["p_pp"] = sess["p_pp"][:1000]
            sess["protocols"] = sess["protocols"][:1000]

        total_sessions += 1
        if sess["label"] == attack_name_norm:
            attack_sessions += 1
        else:
            benign_sessions += 1

        doc = {
            "p_header": sess["p_header"],
            "p_payload": sess["p_payload"],
            "label": sess["label"],
            "total_length": sum(sess["p_tt"]),
            "header_length": sum(sess["p_hh"]),
            "payload_length": sum(sess["p_pp"]),
            "protocols": sess["protocols"][-1] if sess["protocols"] else "Unknown",
            "packet_count": pkt_count,
            "first_ts": sess["first_ts"],
            "last_ts": sess["last_ts"],
            "key": {
                "ip1": key[0],
                "port1": key[1],
                "ip2": key[2],
                "port2": key[3],
                "proto": key[4],
            },
        }

        batch.append(doc)

        # Batch insert
        if len(batch) >= BATCH_SIZE:
            try:
                collection.insert_many(batch, ordered=False)
                batch = []
            except Exception as e:
                logger.error(f"Insert error: {e}")
                batch = []

    logger.info(f"Processing {len(file_paths)} files...")

    # Process files (sorted by filename)
    for file_idx, file_path in enumerate(file_paths, start=1):

        filename = os.path.basename(file_path)
        logger.info(f"Processing file {file_idx}/{len(file_paths)}: {filename}")

        try:
            with PcapReader(file_path) as pcap:
                for pkt in pcap:
                    packets_processed += 1

                    # Reduced GC frequency
                    if packets_processed % GC_INTERVAL == 0:
                        gc.collect()
                        logger.info(
                            f"[{file_idx}/{len(file_paths)}] "
                            f"{packets_processed:,} pkts | "
                            f"{len(active_sessions)} active | "
                            f"{total_sessions:,} total | "
                            f"Attack: {attack_sessions:,}"
                        )

                    # Get timestamp
                    try:
                        ts = float(pkt.time)
                    except:
                        continue

                    # Get 5-tuple key
                    key = get_5tuple_key(pkt)
                    if key is None:
                        continue

                    # Session timeout check
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
                            "label": "Benign-Test",
                        }
                        active_sessions[key] = sess

                    # Process packet (optimized function)
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

                    # Label session
                    label_session_online(sess, key, ts, constraints, attack_name_norm)

        except Exception as e:
            logger.error(f"Error in {os.path.basename(file_path)}: {e}")
            continue

        # *** KEY CHANGE: Flush ALL sessions from this file before moving to next ***
        logger.info(f"  File complete. Flushing {len(active_sessions)} sessions from {filename}...")
        for key, sess in list(active_sessions.items()):
            flush_session(key, sess)

        # Clear the dictionary for next file
        active_sessions.clear()

        logger.info(f"  ✓ {filename}: {packets_processed:,} packets processed")

    # Flush remaining sessions
    logger.info(f"Flushing {len(active_sessions)} remaining sessions...")
    for key, sess in active_sessions.items():
        flush_session(key, sess)

    # Insert final batch
    if batch:
        try:
            collection.insert_many(batch, ordered=False)
        except Exception as e:
            logger.error(f"Final insert error: {e}")

    logger.info("=" * 60)
    logger.info(f"✓ COMPLETE")
    logger.info(f"  Files: {len(file_paths)}")
    logger.info(f"  Packets: {packets_processed:,}")
    logger.info(f"  Sessions: {total_sessions:,}")
    logger.info(f"  Attack: {attack_sessions:,}")
    logger.info(f"  Benign: {benign_sessions:,}")
    logger.info("=" * 60)


def create_indexes(collection):
    """Create indexes."""
    try:
        collection.create_index([("label", 1)], background=True)
        collection.create_index([("first_ts", 1)], background=True)
        collection.create_index([("last_ts", 1)], background=True)
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
    parser = argparse.ArgumentParser(description="Fast PCAP directory processor")
    parser.add_argument("--attack-name", default="dos", help="Attack name")
    parser.add_argument("--csv", default=r"C:\Users\zyf17\Desktop\NOT-IOT\SecurityEvents_Network_datasets\attack_5tuple_time_ranges.csv", help="Constraints CSV")
    parser.add_argument("--pcap-dir", default=r"C:\Users\zyf17\Desktop\NOT-IOT\attack_pcaps\normal_attack_pcaps\normal_DoS", help="PCAP directory")
    parser.add_argument("--mongo-uri", default=MONGO_URI, help="MongoDB URI")
    parser.add_argument("--db-name", default=DB_NAME, help="Database name")
    parser.add_argument("--store-full-bytes", action="store_true", default=False, help="Store full bytes")
    parser.add_argument("--skip-indexes", action="store_true", help="Skip indexes")
    return parser.parse_args()

def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("FAST PCAP PROCESSOR")
    logger.info(f"Attack: {args.attack_name}")
    logger.info(f"Directory: {args.pcap_dir}")
    logger.info("=" * 60)

    # Load constraints
    constraints = load_constraints(args.csv, args.attack_name)
    # Get PCAP files (simple filename sort - NO timestamp scanning)
    pcap_files = [
        os.path.join(args.pcap_dir, f)
        for f in os.listdir(args.pcap_dir)
        if f.endswith(".pcap")
    ]

    if not pcap_files:
        logger.error("No .pcap files found")
        return

    # Simple alphabetical sort
    pcap_files.sort()
    logger.info(f"Found {len(pcap_files)} files (sorted by filename)")

    # Connect to MongoDB
    client = None
    try:
        logger.info("Connecting to MongoDB...")
        client = pymongo.MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        logger.info("✓ Connected")

        db = client[args.db_name]
        coll_name = normalize_attack_name(args.attack_name)
        collection = db[coll_name]

        # Check existing data
        existing = collection.count_documents({})
        if existing > 0:
            logger.warning(f"Collection has {existing} documents")
            response = input("Drop? (y/n): ")
            if response.lower() == 'y':
                db.drop_collection(coll_name)
                collection = db[coll_name]
                logger.info("✓ Dropped")

        # Process
        process_directory_fast(
            file_paths=pcap_files,
            attack_name=args.attack_name,
            constraints=constraints,
            collection=collection,
            store_full_bytes=args.store_full_bytes,
        )

        # Create indexes
        if not args.skip_indexes:
            create_indexes(collection)

        # Final count
        final = collection.count_documents({})
        attack = collection.count_documents({"label": args.attack_name})
        benign = collection.count_documents({"label": "Benign-Test"})

        logger.info("=" * 60)
        logger.info(f"✓ SUCCESS")
        logger.info(f"  Collection: {args.db_name}.{coll_name}")
        logger.info(f"  Total: {final:,}")
        logger.info(f"  Attack: {attack:,}")
        logger.info(f"  Benign: {benign:,}")
        logger.info("=" * 60)

    except pymongo.errors.ServerSelectionTimeoutError:
        logger.error("❌ MongoDB connection failed")
    except KeyboardInterrupt:
        logger.warning("⚠️  Interrupted")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        if client:
            client.close()


if __name__ == '__main__':
    main()