import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from datetime import datetime
import os
import tempfile
import yaml

# Import your pipeline class
from pipeline import ConsumerCreditPipeline

class TestConsumerCreditPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment and SparkSession."""
        cls.spark = SparkSession.builder \
            .appName("TestConsumerCreditPipeline") \
            .master("local[2]") \
            .getOrCreate()
            
        # Create temporary config file
        cls.config = {
            'spark': {'warehouse_dir': '/tmp/spark-warehouse'},
            'required_fields': ['loan_id', 'customer_id', 'loan_amount'],
            'numerical_fields': ['loan_amount', 'interest_rate'],
            'risk_weights': {
                'credit_score': -0.4,
                'debt_to_income_ratio': 0.3
            }
        }
        
        with open('test_config.yml', 'w') as f:
            yaml.dump(cls.config, f)
            
    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        cls.spark.stop()
        if os.path.exists('test_config.yml'):
            os.remove('test_config.yml')

    def setUp(self):
        """Set up test pipeline instance."""
        self.pipeline = ConsumerCreditPipeline(config_path='test_config.yml')
        
        # Create sample test data
        self.test_data = self.spark.createDataFrame([
            ('LOAN001', 'CUST001', 10000.0, 5.0, '2023-01-01', '2024-01-01', 700, 0.3),
            ('LOAN002', 'CUST002', 20000.0, 7.0, '2023-02-01', '2024-02-01', 650, 0.4),
            ('LOAN003', 'CUST003', 15000.0, 6.0, '2023-03-01', '2024-03-01', 800, 0.2),
        ], ['loan_id', 'customer_id', 'loan_amount', 'interest_rate', 'start_date', 
            'end_date', 'credit_score', 'debt_to_income_ratio'])

    def test_data_loading(self):
        """Test data loading functionality."""
        with tempfile.NamedTemporaryFile(suffix='.csv') as tf:
            # Save test data
            self.test_data.toPandas().to_csv(tf.name, index=False)
            
            # Test loading
            loaded_df = self.pipeline.load_data(tf.name)
            
            # Verify data
            self.assertEqual(loaded_df.count(), 3)
            self.assertEqual(len(loaded_df.columns), 8)

    def test_data_validation(self):
        """Test data validation checks."""
        # Test with clean data
        validation_results = self.pipeline.validate_data(self.test_data)
        self.assertTrue(validation_results['validation_status'])
        
        # Test with problematic data
        bad_data = self.test_data.withColumn('loan_amount', self.spark.sql.functions.lit(None))
        validation_results = self.pipeline.validate_data(bad_data)
        self.assertFalse(validation_results['validation_status'])

    def test_data_transformation(self):
        """Test data transformation logic."""
        transformed_df = self.pipeline.transform_data(self.test_data)
        
        # Check new columns exist
        self.assertTrue('risk_score' in transformed_df.columns)
        self.assertTrue('risk_category' in transformed_df.columns)
        self.assertTrue('loan_duration_days' in transformed_df.columns)
        
        # Verify transformations
        results = transformed_df.collect()
        self.assertEqual(len(results), 3)
        
        # Check risk categorization
        categories = set(row.risk_category for row in results)
        self.assertTrue(len(categories) > 0)

    def test_summary_report(self):
        """Test summary report generation."""
        transformed_df = self.pipeline.transform_data(self.test_data)
        summary = self.pipeline.generate_summary_report(transformed_df)
        
        # Verify summary contents
        self.assertEqual(summary['total_loans'], 3)
        self.assertTrue('risk_distribution' in summary)
        self.assertTrue('avg_loan_duration' in summary)
        self.assertTrue('total_loan_value' in summary)

    def test_end_to_end_pipeline(self):
        """Test complete pipeline execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'input.csv')
            output_path = os.path.join(tmpdir, 'output.parquet')
            
            # Save test data
            self.test_data.toPandas().to_csv(input_path, index=False)
            
            # Run pipeline
            summary = self.pipeline.run_pipeline(input_path, output_path)
            
            # Verify output exists and is correct
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue('total_loans' in summary)
            self.assertEqual(summary['total_loans'], 3)

    def test_error_handling(self):
        """Test error handling capabilities."""
        # Test with non-existent file
        with self.assertRaises(Exception):
            self.pipeline.load_data('nonexistent.csv')
            
        # Test with invalid data
        invalid_data = self.spark.createDataFrame([
            ('LOAN001', None, -1000.0)  # Invalid values
        ], ['loan_id', 'customer_id', 'loan_amount'])
        
        validation_results = self.pipeline.validate_data(invalid_data)
        self.assertFalse(validation_results['validation_status'])

if __name__ == '__main__':
    unittest.main()
