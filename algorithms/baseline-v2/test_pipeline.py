#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Turkish Address Normalization Pipeline
"""

import sys
import os

# Add current directory to path to import the main module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_single_address():
    """Test single address normalization."""
    print("Testing single address normalization...")
    
    try:
        # Import the main module
        from untitled13 import preprocess_address
        
        # Test cases
        test_addresses = [
            "Kadıköy Mah. Cadde No:5 K:2 D:4",
            "Beşiktaş Mh. Cd. Apt No5/3",
            "Üsküdar Mahallesi Sokak No:12 Kat 3",
            "Narlıdere İzmir Narlıdere Narlıdere"
        ]
        
        for i, address in enumerate(test_addresses, 1):
            normalized = preprocess_address(address)
            print(f"\nTest {i}:")
            print(f"  Original:  {address}")
            print(f"  Normalized: {normalized}")
            print(f"  Length: {len(address)} → {len(normalized)} chars")
        
        print("\n✅ Single address normalization test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Single address normalization test failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing functionality."""
    print("\nTesting batch processing...")
    
    try:
        import pandas as pd
        from untitled13 import preprocess_dataframe
        
        # Create test data
        test_data = {
            'id': [1, 2, 3, 4, 5],
            'address': [
                "Kadıköy Mah. Cadde No:5 K:2 D:4",
                "Beşiktaş Mh. Cd. Apt No5/3",
                "Üsküdar Mahallesi Sokak No:12 Kat 3",
                "Şişli Mah. Bulvar No:25 Blok A",
                "Fatih Mah. Caddesi No:8 Daire 5"
            ],
            'label': ['A', 'B', 'C', 'D', 'E']
        }
        
        df = pd.DataFrame(test_data)
        print(f"Created test dataframe with {len(df)} addresses")
        
        # Process the data
        processed_df = preprocess_dataframe(df, address_col='address', label_col='label')
        
        print(f"Processing completed. Output shape: {processed_df.shape}")
        print(f"New column 'processed_address' added: {'processed_address' in processed_df.columns}")
        
        # Show results
        print("\nProcessing results:")
        for i, row in processed_df.iterrows():
            print(f"  {i+1}. {row['address']} → {row['processed_address']}")
        
        print("\n✅ Batch processing test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        return False

def test_analysis():
    """Test analysis functionality."""
    print("\nTesting analysis functionality...")
    
    try:
        import pandas as pd
        from untitled13 import preprocess_dataframe, analyze_preprocessing_impact
        
        # Create test data
        test_data = {
            'id': [1, 2, 3, 4, 5],
            'address': [
                "Kadıköy Mah. Cadde No:5 K:2 D:4",
                "Beşiktaş Mh. Cd. Apt No5/3",
                "Üsküdar Mahallesi Sokak No:12 Kat 3",
                "Şişli Mah. Bulvar No:25 Blok A",
                "Fatih Mah. Caddesi No:8 Daire 5"
            ],
            'label': ['A', 'B', 'C', 'D', 'E']
        }
        
        df = pd.DataFrame(test_data)
        
        # Process the data
        processed_df = preprocess_dataframe(df, address_col='address', label_col='label')
        
        # Analyze impact
        analysis = analyze_preprocessing_impact(df, processed_df, 'address')
        
        print("Analysis results:")
        for key, value in analysis.items():
            if isinstance(value, float):
                if 'rate' in key or 'percent' in key:
                    print(f"  {key}: {value:.2%}")
                else:
                    print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n✅ Analysis test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Turkish Address Normalization Pipeline - Test Suite")
    print("=" * 60)
    
    tests = [
        test_single_address,
        test_batch_processing,
        test_analysis
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
