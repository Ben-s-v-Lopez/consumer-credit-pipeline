from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, count, sum, datediff, current_date
import pandas as pd
import logging
import yaml
from datetime import datetime
import os

class ConsumerCreditPipeline:
    """
    Data pipeline for processing consumer credit lending data.
    Demonstrates skills relevant to FCA Data Engineer role:
    - Data pipeline development
    - Python & PySpark
    - Data quality checks
    - Documentation
    """
    
    def __init__(self, config_path='config.yml'):
        """Initialize the pipeline with configuration."""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.spark = self._initialize_spark()
        
    def _setup_logging(self):
        """Configure logging for the pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise

    def _initialize_spark(self):
        """Initialize Spark session."""
        return SparkSession.builder \
            .appName("ConsumerCreditAnalysis") \
            .config("spark.sql.warehouse.dir", self.config['spark']['warehouse_dir']) \
            .getOrCreate()

    def load_data(self, input_path):
        """
        Load consumer credit data from source.
        Handles multiple file formats (CSV, Parquet).
        """
        try:
            file_format = input_path.split('.')[-1]
            if file_format == 'csv':
                df = self.spark.read.csv(input_path, header=True, inferSchema=True)
            elif file_format == 'parquet':
                df = self.spark.read.parquet(input_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            self.logger.info(f"Successfully loaded data from {input_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_data(self, df):
        """
        Perform data quality checks on the input dataset.
        Returns validation results and flagged records.
        """
        validation_results = {
            'total_records': df.count(),
            'null_counts': {},
            'outliers': {},
            'validation_status': True
        }

        # Check for nulls in required fields
        for column in self.config['required_fields']:
            null_count = df.filter(col(column).isNull()).count()
            validation_results['null_counts'][column] = null_count
            if null_count > 0:
                validation_results['validation_status'] = False

        # Check for numerical outliers
        for column in self.config['numerical_fields']:
            stats = df.select(avg(column), 
                            col(column).cast('double').alias('value')) \
                     .agg({'value': 'avg', 'value': 'stddev'}) \
                     .collect()[0]
            
            mean, stddev = stats[0], stats[1]
            outliers = df.filter(
                (col(column) > mean + 3 * stddev) |
                (col(column) < mean - 3 * stddev)
            ).count()
            
            validation_results['outliers'][column] = outliers

        self.logger.info(f"Data validation completed: {validation_results}")
        return validation_results

    def transform_data(self, df):
        """
        Apply business transformations to the data.
        Includes credit risk scoring and loan categorization.
        """
        # Calculate credit risk score based on configured weights
        risk_weights = self.config['risk_weights']
        
        for column, weight in risk_weights.items():
            df = df.withColumn(
                f"{column}_weighted",
                col(column) * weight
            )
        
        # Calculate total risk score
        weighted_columns = [f"{col}_weighted" for col in risk_weights.keys()]
        df = df.withColumn(
            "risk_score",
            sum(col(c) for c in weighted_columns)
        )
        
        # Categorize loans based on risk score
        df = df.withColumn(
            "risk_category",
            when(col("risk_score") <= 30, "Low Risk")
            .when(col("risk_score") <= 70, "Medium Risk")
            .otherwise("High Risk")
        )
        
        # Calculate loan duration
        df = df.withColumn(
            "loan_duration_days",
            datediff(col("end_date"), col("start_date"))
        )
        
        self.logger.info("Data transformation completed")
        return df

    def save_data(self, df, output_path):
        """
        Save processed data to specified location.
        Supports multiple output formats.
        """
        try:
            file_format = output_path.split('.')[-1]
            
            if file_format == 'parquet':
                df.write.mode('overwrite').parquet(output_path)
            elif file_format == 'csv':
                df.write.mode('overwrite').csv(output_path, header=True)
            else:
                raise ValueError(f"Unsupported output format: {file_format}")
                
            self.logger.info(f"Data successfully saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise

    def generate_summary_report(self, df):
        """
        Generate summary statistics and insights from the processed data.
        """
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_loans': df.count(),
            'risk_distribution': df.groupBy('risk_category').count().toPandas().to_dict(),
            'avg_loan_duration': df.select(avg('loan_duration_days')).collect()[0][0],
            'total_loan_value': df.select(sum('loan_amount')).collect()[0][0]
        }
        
        # Save summary report
        with open('summary_report.yml', 'w') as file:
            yaml.dump(summary, file)
        
        self.logger.info("Summary report generated")
        return summary

    def run_pipeline(self, input_path, output_path):
        """
        Execute the complete data pipeline.
        """
        try:
            self.logger.info("Starting pipeline execution")
            
            # Load data
            df = self.load_data(input_path)
            
            # Validate data
            validation_results = self.validate_data(df)
            if not validation_results['validation_status']:
                self.logger.warning("Data validation failed")
                
            # Transform data
            processed_df = self.transform_data(df)
            
            # Save processed data
            self.save_data(processed_df, output_path)
            
            # Generate summary report
            summary = self.generate_summary_report(processed_df)
            
            self.logger.info("Pipeline execution completed successfully")
            return summary
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise
        finally:
            self.spark.stop()

if __name__ == "__main__":
    # Example usage
    pipeline = ConsumerCreditPipeline()
    pipeline.run_pipeline(
        input_path='data/raw/credit_data.csv',
        output_path='data/processed/credit_data_processed.parquet'
    )
