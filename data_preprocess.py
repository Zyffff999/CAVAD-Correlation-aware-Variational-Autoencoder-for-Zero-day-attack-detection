# -*- coding: utf-8 -*-
"""
data_preprocess_simple.py
--------------------------------
Simplified IoT Data Preprocessing
- Pure benign from "benign" collection only
- Mixed benign for training (random, no filter)
- Test set: pure benign + filtered attacks only
"""

import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import pymongo


# =========================
# Configuration
# =========================
@dataclass
class Config:
    """Configuration for data preprocessing"""

    # Database
    db_host: str = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    db_name: str = "NOT_IOT-1"

    # Data Dimensions
    max_header_size: int = 128
    max_payload_size: int = 128
    max_packets: int = 16

    # Pure Benign Settings
    pure_benign_collection: str = "benign"  # Fixed collection name
    pure_benign_train_ratio: float = 0.65  # 70% for training (applied per stratification bucket)
    test_benign_max_samples: int = 15000  # Max test benign samples (None = use all, e.g., 10000)

    # Mixed Benign for Training (random noise, no filter)
    mixed_benign_samples_per_collection: int = 5000  # Per attack collection

    # Attack Settings for Testing (filter in MongoDB, then sample)
    attack_min_packet_count: int = 1  # Filter criteria
    attack_min_payload_length: int = 0  # Filter criteria
    attack_target_per_collection: int = 1000  # Target to sample after filter

    # Labels
    mixed_benign_label: str = "Benign-Test"

    # Output
    data_dir: str = r"data/NOT-IOT"

    @property
    def subdir(self) -> str:
        return f"Session-{self.max_packets}-h{self.max_header_size}-p{self.max_payload_size}"


CFG = Config()
# =========================
# Helper Functions
# =========================
def hex_to_bytes(hex_string: str) -> List[int]:
    """Convert hex string to list of integers"""
    if not hex_string:
        return []
    s = hex_string.replace(" ", "").replace("\n", "")
    if len(s) % 2 != 0:
        s = s[:-1]
    try:
        return [int(s[i:i + 2], 16) for i in range(0, len(s), 2)]
    except ValueError:
        return []


def pad_or_truncate(arr: List[int], target_len: int) -> List[int]:
    """Pad with zeros or truncate to target length"""
    if len(arr) >= target_len:
        return arr[:target_len]
    return arr + [0] * (target_len - len(arr))


def calculate_total_payload_length(doc: Dict) -> int:
    """Calculate total payload bytes across all packets"""
    payloads = doc.get("p_payload", []) or []
    total = 0
    for payload_hex in payloads:
        if payload_hex:
            s = payload_hex.replace(" ", "").replace("\n", "")
            total += len(s) // 2
    return total


def get_stratification_key(doc: Dict) -> str:
    """Generate stratification key based on protocol, payload size, and session pattern"""
    # Protocol
    proto = "OTHER"
    if "key" in doc and "proto" in doc["key"]:
        proto = str(doc["key"]["proto"]).upper()
    elif "protocols" in doc and len(doc["protocols"]) > 0:
        proto = str(doc["protocols"][0]).upper()

    if "TCP" in proto:
        proto = "TCP"
    elif "UDP" in proto:
        proto = "UDP"
    else:
        proto = "OTH"

    # Payload size bucketing
    total_payload = calculate_total_payload_length(doc)

    if total_payload == 0:
        payload_bucket = "ZERO"
    elif total_payload <= 50:
        payload_bucket = "TINY"
    elif total_payload <= 500:
        payload_bucket = "SMALL"
    elif total_payload <= 2000:
        payload_bucket = "MEDIUM"
    else:
        payload_bucket = "LARGE"

    # Session pattern
    pkt_count = doc.get("packet_count", 0)

    if pkt_count <= 2:
        session_pattern = "INCOMPLETE"
    elif pkt_count <= 5:
        session_pattern = "SHORT"
    elif pkt_count <= 12:
        session_pattern = "NORMAL"
    else:
        session_pattern = "LONG"

    return f"{proto}_{payload_bucket}_{session_pattern}"


def process_session(doc: Dict, label: str) -> Tuple:
    """Process a single session document"""
    headers = doc.get("p_header", []) or []
    payloads = doc.get("p_payload", []) or []

    n = min(len(headers), len(payloads), CFG.max_packets)

    H = np.zeros((CFG.max_packets, CFG.max_header_size), dtype=np.uint8)
    P = np.zeros((CFG.max_packets, CFG.max_payload_size), dtype=np.uint8)
    M = np.zeros(CFG.max_packets, dtype=bool)

    for i in range(n):
        H[i, :] = pad_or_truncate(hex_to_bytes(headers[i]), CFG.max_header_size)
        P[i, :] = pad_or_truncate(hex_to_bytes(payloads[i]), CFG.max_payload_size)
        M[i] = (len(payloads[i]) > 0)

    stats = np.zeros(20, dtype=np.float32)

    return (H, P, M, label, stats)


def filter_attack(doc: Dict) -> bool:
    """Filter for attack data"""
    pkt_count = doc.get("packet_count", 0)
    if pkt_count < CFG.attack_min_packet_count:
        return False

    payload_len = doc.get("payload_length", 0)
    if payload_len <= CFG.attack_min_payload_length:
        return False

    return True


# =========================
# Data Collection
# =========================
def collect_pure_benign(db) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Collect pure benign from 'benign' collection
    Stratify by traffic pattern and split each bucket by train/test ratio
    """
    print(f"\n{'=' * 70}")
    print(f"📦 Collecting Pure Benign (from '{CFG.pure_benign_collection}' collection)")
    print(f"{'=' * 70}")
    print(
        f"Strategy: Stratify → Split each bucket {CFG.pure_benign_train_ratio:.0%}/{1 - CFG.pure_benign_train_ratio:.0%}")

    collection = db[CFG.pure_benign_collection]
    total_docs = collection.count_documents({})

    # Step 1: Collect all and group by stratification key
    buckets = defaultdict(list)

    with tqdm(total=total_docs, desc="Reading & Stratifying", unit="doc") as pbar:
        for doc in collection.find({}):
            strat_key = get_stratification_key(doc)
            processed = process_session(doc, "Benign")
            buckets[strat_key].append(processed)
            pbar.update(1)

    print(f"\n✅ Collected {total_docs:,} samples into {len(buckets)} stratification buckets")

    # Step 2: Split each bucket by ratio
    train_pool = []
    test_pool = []

    print(f"\n📊 Per-Bucket Train/Test Split:")
    print(f"{'Bucket':<35} {'Total':>8} {'Train':>8} {'Test':>8}")
    print(f"{'-' * 70}")

    for strat_key in sorted(buckets.keys()):
        samples = buckets[strat_key]
        random.shuffle(samples)

        split_idx = int(len(samples) * CFG.pure_benign_train_ratio)
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]

        train_pool.extend(train_samples)
        test_pool.extend(test_samples)

        print(f"{strat_key:<35} {len(samples):>8,} {len(train_samples):>8,} {len(test_samples):>8,}")

    print(f"{'-' * 70}")
    print(f"{'TOTAL':<35} {total_docs:>8,} {len(train_pool):>8,} {len(test_pool):>8,}")

    # Apply test benign max samples limit if configured
    if CFG.test_benign_max_samples and len(test_pool) > CFG.test_benign_max_samples:
        print(f"\n⚠️  Test benign exceeds limit: {len(test_pool):,} > {CFG.test_benign_max_samples:,}")
        print(f"   Randomly sampling {CFG.test_benign_max_samples:,} samples...")
        random.shuffle(test_pool)
        test_pool = test_pool[:CFG.test_benign_max_samples]
        print(f"   ✅ Test benign limited to: {len(test_pool):,}")

    return train_pool, test_pool


def collect_mixed_benign_for_training(db, attack_collections: List[str]) -> List[Tuple]:
    """Randomly collect mixed benign from attack collections (NO FILTER)"""
    print(f"\n{'=' * 70}")
    print(f"📦 Collecting Mixed Benign for Training (Random, No Filter)")
    print(f"{'=' * 70}")
    print(f"Target: {CFG.mixed_benign_samples_per_collection} per collection")
    print(f"Collections: {len(attack_collections)}")

    pool = []

    for coll in tqdm(attack_collections, desc="Collecting Mixed Benign", unit="coll"):
        try:
            # Query for Benign-Test label
            query = {"$or": [
                {"label": CFG.mixed_benign_label},
                {"Label": CFG.mixed_benign_label}
            ]}

            available = db[coll].count_documents(query)

            if available == 0:
                continue

            # Random sample (no filter)
            sample_size = min(available, CFG.mixed_benign_samples_per_collection)

            pipeline = [
                {"$match": query},
                {"$sample": {"size": sample_size}}
            ]

            cursor = db[coll].aggregate(pipeline, allowDiskUse=True)

            for doc in cursor:
                pool.append(process_session(doc, "Benign"))

        except Exception as e:
            tqdm.write(f"⚠️  {coll}: {str(e)}")

    print(f"✅ Collected: {len(pool):,} mixed benign samples")
    return pool


def collect_attacks_for_testing(db, attack_collections: List[str]) -> Tuple[List[Tuple], Dict]:
    """Collect attacks: filter in MongoDB, then sample 1000"""
    print(f"\n{'=' * 70}")
    print(f"📦 Collecting Attacks for Testing")
    print(f"{'=' * 70}")
    print(f"Strategy: Filter in MongoDB → Sample {CFG.attack_target_per_collection}")
    print(f"Filter: packet_count >= {CFG.attack_min_packet_count}, "
          f"payload_length > {CFG.attack_min_payload_length}")

    pool = []
    stats = {
        'total_available': 0,
        'total_selected': 0,
        'collections': []
    }

    for coll in tqdm(attack_collections, desc="Collecting Attacks", unit="coll"):
        try:
            # Build filter query (filter in MongoDB - much faster!)
            query = {
                "$nor": [
                    {"label": CFG.mixed_benign_label},
                    {"Label": CFG.mixed_benign_label}
                ],
                "packet_count": {"$gte": CFG.attack_min_packet_count},
                "payload_length": {"$gt": CFG.attack_min_payload_length}
            }

            # Count how many pass the filter
            available = db[coll].count_documents(query)

            if available == 0:
                stats['collections'].append({
                    'name': coll,
                    'available': 0,
                    'selected': 0
                })
                continue

            # Sample from filtered results
            sample_size = min(available, CFG.attack_target_per_collection)

            pipeline = [
                {"$match": query},
                {"$sample": {"size": sample_size}}
            ]

            cursor = db[coll].aggregate(pipeline, allowDiskUse=True)

            # Process selected samples
            selected_count = 0
            for doc in cursor:
                pool.append(process_session(doc, coll))
                selected_count += 1

            stats['total_available'] += available
            stats['total_selected'] += selected_count

            stats['collections'].append({
                'name': coll,
                'available': available,
                'selected': selected_count
            })

        except Exception as e:
            tqdm.write(f"❌ {coll}: {str(e)}")
            stats['collections'].append({
                'name': coll,
                'available': 'ERROR',
                'selected': 0
            })

    print(f"\n✅ Attack Collection Summary:")
    print(f"{'Collection':<45} {'Available':>10} {'Sampled':>10} {'Status':>10}")
    print(f"{'-' * 78}")

    full_count = 0
    partial_count = 0
    skipped_count = 0

    for coll_stat in stats['collections']:
        name = coll_stat['name'][:43]
        available = coll_stat['available']
        selected = coll_stat['selected']

        # Determine status
        if selected == CFG.attack_target_per_collection:
            status = 'FULL'
            indicator = '✅'
            full_count += 1
        elif selected > 0:
            status = 'PARTIAL'
            indicator = '⚠️ '
            partial_count += 1
        else:
            status = 'SKIPPED'
            indicator = '⏭️ '
            skipped_count += 1

        # Format available (handle ERROR case)
        avail_str = str(available)[:8] if isinstance(available, int) else 'ERROR'

        print(f"{indicator} {name:<43} {avail_str:>10} {selected:>10,} {status:>10}")

    print(f"{'-' * 78}")
    print(f"Summary:")
    print(f"  Full Quota:     {full_count:>4} collections (got {CFG.attack_target_per_collection})")
    print(f"  Partial:        {partial_count:>4} collections (got < {CFG.attack_target_per_collection})")
    print(f"  Skipped:        {skipped_count:>4} collections (no data)")
    print(f"  Total Selected: {stats['total_selected']:>6,} attacks")

    return pool, stats


# =========================
# Output Functions
# =========================
def write_npz(data_list: List[Tuple], npz_path: str, label_map: Dict[str, int]) -> List[int]:
    """Write processed data to NPZ file"""
    if not data_list:
        return []

    print(f"\n💾 Saving {os.path.basename(npz_path)} ({len(data_list):,} samples)...")

    H = np.stack([x[0] for x in data_list]).astype(np.uint8)
    P = np.stack([x[1] for x in data_list]).astype(np.uint8)
    M = np.stack([x[2] for x in data_list]).astype(np.bool_)
    S = np.stack([x[4] for x in data_list]).astype(np.float32)

    labels = []
    for item in data_list:
        raw_label = item[3]

        if raw_label == "Benign":
            label_id = 0
        else:
            if raw_label not in label_map:
                label_map[raw_label] = len(label_map)
            label_id = label_map[raw_label]

        labels.append(label_id)

    L = np.array(labels, dtype=np.int32)

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(
        npz_path,
        headers=H,
        payloads=P,
        payload_masks=M,
        labels=L,
        stats_features=S
    )

    print(f"✅ Saved to {npz_path}")
    return labels


def save_label_map(out_dir: str, label_map: Dict[str, int]):
    """Save separate label_map.json file"""
    # Convert to lowercase for labels
    label_to_id = {}
    for label, idx in label_map.items():
        # Convert "Benign" to "benign", attack names to lowercase
        label_lower = label.lower()
        label_to_id[label_lower] = idx

    # Create reverse mapping (id to label)
    id_to_label = {str(idx): label for label, idx in label_to_id.items()}

    label_map_data = {
        "label_to_id": label_to_id,
        "id_to_label": id_to_label
    }

    label_map_path = os.path.join(out_dir, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump(label_map_data, f, indent=2)

    print(f"✅ Label map saved to: {label_map_path}")

    # Print summary
    print(f"\n📋 Label Mapping ({len(label_to_id)} classes):")
    for label, idx in sorted(label_to_id.items(), key=lambda x: x[1]):
        print(f"   {idx:>3}: {label}")


def save_metadata(out_dir: str, label_map: Dict[str, int],
                  train_stats: Dict, test_stats: Dict):
    """Save metadata"""
    metadata = {
        'configuration': {
            'max_packets': CFG.max_packets,
            'max_header_size': CFG.max_header_size,
            'max_payload_size': CFG.max_payload_size,
            'pure_benign_train_ratio': CFG.pure_benign_train_ratio,
            'mixed_benign_samples_per_collection': CFG.mixed_benign_samples_per_collection,
            'attack_target_per_collection': CFG.attack_target_per_collection,
        },
        'label_map': label_map,
        'dataset_statistics': {
            'train': train_stats,
            'test': test_stats
        }
    }

    metadata_path = os.path.join(out_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved to: {metadata_path}")


# =========================
# Main Pipeline
# =========================
def main():
    """Main execution pipeline"""
    print(f"\n{'#' * 70}")
    print(f"#  SIMPLIFIED DATA PREPROCESSING PIPELINE")
    print(f"#  Database: {CFG.db_name}")
    print(f"{'#' * 70}")

    # Connect to MongoDB
    client = pymongo.MongoClient(CFG.db_host)
    db = client[CFG.db_name]
    all_collections = sorted(db.list_collection_names())

    print(f"\n🔌 Connected to MongoDB: {len(all_collections)} collections")

    # Identify collections
    pure_benign_coll = CFG.pure_benign_collection
    attack_collections = [c for c in all_collections if c != pure_benign_coll]

    print(f"   Pure Benign: '{pure_benign_coll}'")
    print(f"   Attack Collections: {len(attack_collections)}")

    # ==================
    # STEP 1: Pure Benign (stratified split per bucket)
    # ==================
    train_pure, test_pure = collect_pure_benign(db)

    print(f"\n📊 Pure Benign Stratified Split:")
    print(f"   Training: {len(train_pure):,}")
    print(f"   Testing:  {len(test_pure):,}")

    # ==================
    # STEP 2: Mixed Benign for Training (random, no filter)
    # ==================
    train_mixed = collect_mixed_benign_for_training(db, attack_collections)

    # ==================
    # STEP 3: Attacks for Testing (sample 1500, filter, select 1000)
    # ==================
    test_attacks, attack_stats = collect_attacks_for_testing(db, attack_collections)

    # ==================
    # STEP 4: Assemble Final Datasets
    # ==================
    print(f"\n{'=' * 70}")
    print(f"🔨 ASSEMBLING FINAL DATASETS")
    print(f"{'=' * 70}")

    # Training: Pure Benign + Mixed Benign (for noise)
    final_train = train_pure + train_mixed
    random.shuffle(final_train)

    # Testing: Pure Benign + Attacks (NO mixed benign)
    final_test = test_pure + test_attacks
    random.shuffle(final_test)

    print(f"\nTraining Set: {len(final_train):,} samples")
    print(f"   ├─ Pure Benign:  {len(train_pure):,}")
    print(f"   └─ Mixed Benign: {len(train_mixed):,} (noise)")

    print(f"\nTest Set: {len(final_test):,} samples")
    print(f"   ├─ Pure Benign: {len(test_pure):,}")
    print(f"   └─ Attacks:     {len(test_attacks):,}")

    # ==================
    # STEP 5: Save
    # ==================
    print(f"\n{'=' * 70}")
    print(f"💾 SAVING DATA")
    print(f"{'=' * 70}")

    label_map = {"Benign": 0}
    out_dir = os.path.join(CFG.data_dir, CFG.subdir)

    write_npz(final_train, os.path.join(out_dir, "train.npz"), label_map)
    write_npz(final_test, os.path.join(out_dir, "test.npz"), label_map)

    train_stats = {
        'pure_benign': len(train_pure),
        'mixed_benign': len(train_mixed),
        'total': len(final_train)
    }

    test_stats = {
        'pure_benign': len(test_pure),
        'attacks': len(test_attacks),
        'total': len(final_test),
        'attack_details': attack_stats
    }

    save_metadata(out_dir, label_map, train_stats, test_stats)
    save_label_map(out_dir, label_map)

    print(f"\n✅ Pipeline completed!")
    print(f"📁 Output: {out_dir}")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()