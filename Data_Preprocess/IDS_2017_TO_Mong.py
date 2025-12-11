import sys
import os
import pymongo
from tqdm import tqdm
from collections import Counter
from abc import ABCMeta, abstractmethod
import string
import socket
import struct
import gc
import random
import numpy as np
from scapy.all import *
from scapy.utils import hexdump
import logging
from datetime import datetime

# Configuration constants
MIN_PACKET_COUNT = 1
MAX_PACKET_COUNT = 50
GC_INTERVAL = 1000
BATCH_SIZE = 1000
MAX_FILES_TO_PROCESS = 400000

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def bytes2str(info):
    """Convert bytes or byte array to hex string."""
    if isinstance(info, bytes):
        return info.hex()
    else:
        return bytes(info).hex()


def get_payload_bytes(packet):
    """Extract payload bytes from packet."""
    if packet.haslayer("Raw"):
        payload_bytes = list(packet["Raw"].load)
        return payload_bytes
    else:
        return []


def get_header_and_payload_bytes(packet):
    """Extract header and payload bytes from packet."""
    header_bytes = []

    if packet.haslayer("Ether"):
        header_bytes.extend(bytes(packet["Ether"]))
    if packet.haslayer("IP"):
        header_bytes.extend(bytes(packet["IP"]))
    if packet.haslayer("TCP"):
        header_bytes.extend(bytes(packet["TCP"]))
    elif packet.haslayer("UDP"):
        header_bytes.extend(bytes(packet["UDP"]))

    payload_bytes = get_payload_bytes(packet)
    return header_bytes, payload_bytes


def mask_packet_addresses(packet):
    """Mask MAC and IP addresses for privacy."""
    pkt = packet.copy()

    if pkt.haslayer("Ether"):
        pkt["Ether"].src = "00:00:00:00:00:00"
        pkt["Ether"].dst = "00:00:00:00:00:00"

    if pkt.haslayer("IP"):
        pkt["IP"].src = "0.0.0.0"
        pkt["IP"].dst = "0.0.0.0"

    return pkt


def get_protocol(pkt):
    if pkt.haslayer("TCP"):
        return 'TCP'
    elif pkt.haslayer("UDP"):
        return 'UDP'
    elif pkt.haslayer("ICMP"):
        return 'ICMP'
    elif pkt.haslayer("ARP"):
        return 'ARP'
    elif pkt.haslayer("IPv6"):
        return 'IPv6'
    elif pkt.haslayer("DNS"):
        return 'DNS'
    elif pkt.haslayer("HTTP"):
        return 'HTTP'
    elif pkt.haslayer("HTTPS"):
        return 'HTTPS'
    else:
        return 'Other'


def process_packet(pkt):
    pkt = mask_packet_addresses(pkt)
    p_header, p_payload = get_header_and_payload_bytes(pkt)

    total_length = len(bytes(pkt))
    header_length = len(p_header)
    payload_length = len(p_payload)

    protocol = get_protocol(pkt)

    p_payload = bytes2str(p_payload)
    p_header = bytes2str(p_header)

    return p_header, p_payload, total_length, header_length, payload_length, protocol, bytes2str(pkt)


def pcap_to_mongodb(dir_path_dict, label, collection, store_full_bytes=False):
    if not dir_path_dict:
        logger.warning("No PCAP files to process")
        return 0

    batch = []
    sessions_processed = 0
    files_processed = 0

    for idx, dir_key in enumerate(tqdm(dir_path_dict, desc="Processing PCAPs", dynamic_ncols=True)):
        # Garbage collection at intervals
        if (idx + 1) % GC_INTERVAL == 0:
            gc.collect()
            logger.debug(f"Performed garbage collection after {idx + 1} files")

        # Initialize session data
        p_header = []
        p_payload = []
        p_tt = []
        p_hh = []
        p_pp = []
        protocols = []
        bytes_all = []

        session = dict()
        file_path = dir_path_dict[dir_key]

        try:
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                continue

            if not os.access(file_path, os.R_OK):
                logger.error(f"No read permission for file: {file_path}")
                continue

            packet_count = 0
            with PcapReader(file_path) as pcap:
                for pkt in pcap:
                    try:
                        p_header_tmp, p_payload_tmp, p_tt_tmp, p_hh_tmp, p_pp_tmp, protocol, byte_all = process_packet(
                            pkt)

                        p_header.append(p_header_tmp)
                        p_payload.append(p_payload_tmp)
                        p_tt.append(p_tt_tmp)
                        p_hh.append(p_hh_tmp)
                        p_pp.append(p_pp_tmp)
                        protocols.append(protocol)

                        if store_full_bytes:
                            bytes_all.append(byte_all)

                        packet_count += 1

                    except Exception as e:
                        logger.debug(f"Error processing packet {packet_count} in {file_path}: {str(e)}")
                        continue

            # Build session document
            session["p_header"] = p_header
            session["p_payload"] = p_payload
            session['label'] = label
            session["total_length"] = sum(p_tt)
            session["header_length"] = sum(p_hh)
            session["payload_length"] = sum(p_pp)
            session['protocols'] = protocols
            session['dominant_protocol'] = max(set(protocols), key=protocols.count) if protocols else 'Unknown'
            session['packet_count'] = len(p_tt)
            session['filename'] = dir_key
            session['timestamp'] = datetime.utcnow()

            if store_full_bytes:
                session['bytes'] = bytes_all

            if MIN_PACKET_COUNT <= session['packet_count'] <= MAX_PACKET_COUNT:
                batch.append(session)
                if len(batch) >= BATCH_SIZE:
                    try:
                        collection.insert_many(batch)
                        sessions_processed += len(batch)
                        logger.info(f"Inserted batch of {len(batch)} sessions")
                        batch = []
                    except Exception as e:
                        logger.error(f"Error inserting batch: {str(e)}")
                        batch = []
            else:
                logger.debug(
                    f"Skipped {dir_key}: packet count {session['packet_count']} outside range [{MIN_PACKET_COUNT}, {MAX_PACKET_COUNT}]")

            files_processed += 1

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except PermissionError:
            logger.error(f"Permission denied: {file_path}")
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {str(e)}")

    if batch:
        try:
            collection.insert_many(batch)
            sessions_processed += len(batch)
            logger.info(f"Inserted final batch of {len(batch)} sessions")
        except Exception as e:
            logger.error(f"Error inserting final batch: {str(e)}")

    logger.info(f"Processing complete: {files_processed} files processed, {sessions_processed} sessions stored")
    return sessions_processed


def create_dir_path_dict(dir_path, num=MAX_FILES_TO_PROCESS):
    """
    Create a dictionary mapping filenames to their full paths.

    Args:
        dir_path: Path to directory or single PCAP file
        num: Maximum number of files to process

    Returns:
        dict: Mapping of filename to full path
    """
    dir_path_dict = {}
    if os.path.isfile(dir_path) and dir_path.endswith('.pcap'):
        # Single file
        filename = os.path.basename(dir_path)
        dir_path_dict[filename] = dir_path
        logger.info(f"Added single file: {filename}")

    elif os.path.isdir(dir_path):
        try:
            pcap_files = [
                (filename, os.path.join(dir_path, filename), os.path.getsize(os.path.join(dir_path, filename)))
                for filename in os.listdir(dir_path)
                if filename.endswith('.pcap')
            ]

            for idx, (filename, file_path, file_size) in enumerate(pcap_files[:num]):
                dir_path_dict[filename] = file_path

            logger.info(f"Found {len(pcap_files)} PCAP files, processing {len(dir_path_dict)}")

        except Exception as e:
            logger.error(f"Error reading directory {dir_path}: {str(e)}")
    else:
        logger.error(f"Invalid path: {dir_path}")

    return dir_path_dict


def main():

    host = 'mongodb://localhost:27017/'
    db_name = 'IDS_2017'
    data_dir = r"D:\OOD_detect\data\IDS2017"
    client = None
    try:
        logger.info(f"Connecting to MongoDB at {host}")
        client = pymongo.MongoClient(host)
        client.server_info()
        logger.info("Successfully connected to MongoDB")

        db = client[db_name]
        items_to_process = [
            item for item in os.listdir(data_dir)
            if item not in ["csv", "pcap"] and os.path.isdir(os.path.join(data_dir, item))
        ]
        logger.info(f"Found {len(items_to_process)} directories to process")

        for fn in items_to_process:
            pcap_path = os.path.join(data_dir, fn)
            collection_name = os.path.basename(pcap_path)

            logger.info(f"\nProcessing collection: {collection_name}")
            collection = db[collection_name]
            existing_count = collection.count_documents({})
            if existing_count > 0:
                continue

            dir_path_dict = create_dir_path_dict(pcap_path)
            if dir_path_dict:
                sessions_processed = pcap_to_mongodb(
                    dir_path_dict,
                    collection_name,
                    collection,
                    store_full_bytes=True
                )

                logger.info(f"Completed processing {collection_name}: {sessions_processed} sessions added")
            else:
                logger.warning(f"No PCAP files found in {pcap_path}")

    except pymongo.errors.ServerSelectionTimeoutError:
        logger.error("Failed to connect to MongoDB. Please ensure MongoDB is running.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        if client:
            client.close()
            logger.info("Closed MongoDB connection")

if __name__ == '__main__':
    main()