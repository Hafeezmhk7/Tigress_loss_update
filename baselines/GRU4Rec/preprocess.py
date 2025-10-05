#!/usr/bin/env python3
"""
Fixed preprocessing script for Beauty reviews data to convert it to GRU4Rec format.
Converts from: user_id item_id rating timestamp
To: SessionId ItemId Time (tab-separated with header)
"""

import pandas as pd
import numpy as np
import argparse
import os

def preprocess_beauty_data_fixed(input_file, output_file, min_session_length=2, min_item_support=5):
    """
    Preprocess beauty reviews data for GRU4Rec with proper time handling.
    
    Parameters:
    -----------
    input_file : str
        Path to input file (format: user_id item_id rating timestamp)
    output_file : str
        Path to output file (format: SessionId ItemId Time)
    min_session_length : int
        Minimum number of items per session to keep
    min_item_support : int
        Minimum number of times an item must appear to keep it
    """
    
    print(f"Loading data from {input_file}")
    
    # Load data - assuming space-separated format from the beauty reviews
    data = pd.read_csv(input_file, sep=' ', header=None, 
                      names=['SessionId', 'ItemId', 'Rating', 'Time'])
    
    print(f"Original data shape: {data.shape}")
    print(f"Number of unique sessions: {data['SessionId'].nunique()}")
    print(f"Number of unique items: {data['ItemId'].nunique()}")
    print(f"Time range: {data['Time'].min()} to {data['Time'].max()}")
    
    # Filter out items with low support
    item_support = data['ItemId'].value_counts()
    popular_items = item_support[item_support >= min_item_support].index
    data = data[data['ItemId'].isin(popular_items)]
    
    print(f"After filtering items with support < {min_item_support}: {data.shape}")
    print(f"Number of unique items: {data['ItemId'].nunique()}")
    
    # Filter out sessions with too few items
    session_lengths = data.groupby('SessionId').size()
    good_sessions = session_lengths[session_lengths >= min_session_length].index
    data = data[data['SessionId'].isin(good_sessions)]
    
    print(f"After filtering sessions with length < {min_session_length}: {data.shape}")
    print(f"Number of unique sessions: {data['SessionId'].nunique()}")
    
    # Sort by SessionId and Time to ensure proper ordering
    data = data.sort_values(['SessionId', 'Time']).reset_index(drop=True)
    
    # Convert SessionId to integer (may be string)
    session_id_map = {sid: i for i, sid in enumerate(data['SessionId'].unique())}
    data['SessionId'] = data['SessionId'].map(session_id_map)
    
    # Convert item IDs to integers starting from 1
    item_id_map = {iid: i+1 for i, iid in enumerate(data['ItemId'].unique())}
    data['ItemId'] = data['ItemId'].map(item_id_map)
    
    # Create sequential time within each session (1, 2, 3, ...)
    # This ensures proper temporal ordering for GRU4Rec
    data['Time'] = data.groupby('SessionId').cumcount() + 1
    
    # Keep only required columns in the correct order
    processed_data = data[['SessionId', 'ItemId', 'Time']].copy()
    
    print(f"Final data shape: {processed_data.shape}")
    print(f"Final number of unique sessions: {processed_data['SessionId'].nunique()}")
    print(f"Final number of unique items: {processed_data['ItemId'].nunique()}")
    print(f"Time range after processing: {processed_data['Time'].min()} to {processed_data['Time'].max()}")
    
    # Save processed data
    print(f"Saving processed data to {output_file}")
    processed_data.to_csv(output_file, sep='\t', index=False)
    
    # Create train/test split (80/20 by sessions)
    unique_sessions = processed_data['SessionId'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_sessions)
    
    train_sessions = unique_sessions[:int(0.8 * len(unique_sessions))]
    test_sessions = unique_sessions[int(0.8 * len(unique_sessions)):]
    
    train_data = processed_data[processed_data['SessionId'].isin(train_sessions)]
    test_data = processed_data[processed_data['SessionId'].isin(test_sessions)]
    
    train_file = output_file.replace('.txt', '_train.txt')
    test_file = output_file.replace('.txt', '_test.txt')
    
    train_data.to_csv(train_file, sep='\t', index=False)
    test_data.to_csv(test_file, sep='\t', index=False)
    
    print(f"Train data saved to {train_file} ({train_data.shape[0]} interactions)")
    print(f"Test data saved to {test_file} ({test_data.shape[0]} interactions)")
    
    # Show sample data
    print("\nSample of processed data:")
    print(processed_data.head(10))
    
    return processed_data, train_data, test_data

def main():
    parser = argparse.ArgumentParser(description='Fixed preprocessing for Beauty reviews data for GRU4Rec')
    parser.add_argument('--input', '-i', default='data/reviews_Beauty_5.txt', 
                       help='Input file path (default: data/reviews_Beauty_5.txt)')
    parser.add_argument('--output', '-o', default='data/beauty_processed.txt',
                       help='Output file path (default: data/beauty_processed.txt)')
    parser.add_argument('--min_session_length', type=int, default=2,
                       help='Minimum session length (default: 2)')
    parser.add_argument('--min_item_support', type=int, default=5,
                       help='Minimum item support (default: 5)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Preprocess data
    preprocess_beauty_data_fixed(args.input, args.output, 
                                args.min_session_length, args.min_item_support)

if __name__ == '__main__':
    main()