import numpy as np
import os
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import uuid
from decimal import Decimal


class Benchmark:
    def __init__(self, providers, num_requests, models, prompt, streaming=False):
        self.providers = providers
        self.num_requests = num_requests
        self.models = models
        self.prompt = prompt
        self.streaming = streaming
        self.run_id = str(uuid.uuid4())  # Generate a unique ID for each benchmark run

        # Initialize DynamoDB
        self.dynamodb = boto3.resource(
            'dynamodb',
            endpoint_url=os.getenv("DYNAMODB_ENDPOINT_URL"),
            region_name=os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        self.table_name = "BenchmarkMetrics"  # Replace with your DynamoDB table name

        # Data structure to hold all metrics for this run
        self.benchmark_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": self.prompt,
            "providers": {}
        }

    def clean_data(self, data):
        """
        Recursively removes any empty values (None, empty strings, empty lists, etc.)
        from a dictionary to ensure compatibility with DynamoDB.
        """
        if isinstance(data, dict):
            return {k: self.clean_data(v) for k, v in data.items() if v not in [None, '', [], {}]}
        elif isinstance(data, list):
            return [self.clean_data(v) for v in data if v not in [None, '', [], {}]]
        elif isinstance(data, float):
            return Decimal(str(data))  # Convert float to Decimal for DynamoDB
        return data

    def store_data_points(self):
        """
        Store the complete benchmark data in DynamoDB for future visualization.
        """
        table = self.dynamodb.Table(self.table_name)
        
        # Clean the data before storing
        clean_benchmark_data = self.clean_data(self.benchmark_data)
        
        # Print for debugging
        print("Storing the following data in DynamoDB:", clean_benchmark_data)
        
        try:
            table.put_item(Item=clean_benchmark_data)
            print(f"Stored benchmark data for run ID {self.run_id}")
        except ClientError as e:
            print(f"Error saving to DynamoDB: {e.response['Error']['Message']}")

    def add_metric_data(self, provider_name, model_name, metric, latencies):
        """
        Add latency and CDF data to the benchmark data structure.

        Args:
            provider_name (str): The name of the provider.
            model_name (str): The name of the model.
            metric (str): The metric type (e.g., response_times, timetofirsttoken).
            latencies (list): List of sorted latency values in milliseconds.
        """
        # Calculate CDF
        latencies_sorted = np.sort(latencies) * 1000  # Convert to milliseconds
        cdf = np.arange(1, len(latencies_sorted) + 1) / len(latencies_sorted)

        # Convert floats to Decimal for DynamoDB compatibility
        latencies_sorted = [Decimal(str(val)) for val in latencies_sorted.tolist()]
        cdf = [Decimal(str(val)) for val in cdf.tolist()]

        # Initialize provider and model entries if not already present
        if provider_name not in self.benchmark_data["providers"]:
            self.benchmark_data["providers"][provider_name] = {}
        if model_name not in self.benchmark_data["providers"][provider_name]:
            self.benchmark_data["providers"][provider_name][model_name] = {}

        # Add metric data
        self.benchmark_data["providers"][provider_name][model_name][metric] = {
            "latencies": latencies_sorted,
            "cdf": cdf
        }

    def plot_metrics(self, metric):
        """
        Collects latency and CDF data points for each provider and model,
        and adds them to the benchmark data structure instead of plotting.
        
        Args:
            metric (str): The metric type to plot (e.g., response_times).
        """
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            for model, latencies in provider.metrics[metric].items():
                model_name = provider.get_model_name(model)
                self.add_metric_data(provider_name, model_name, metric, latencies)

    def run(self):
        """
        Run the benchmark and store metrics in DynamoDB.
        """
        for provider in self.providers:
            for model in self.models:
                for _ in range(self.num_requests):
                    if self.streaming:
                        provider.perform_inference_streaming(model, self.prompt)
                    else:
                        provider.perform_inference(model, self.prompt)

        # Collect and store metrics in the data structure
        if not self.streaming:
            self.plot_metrics("response_times")
        else:
            self.plot_metrics("timetofirsttoken")
            self.plot_metrics("response_times")
            self.plot_metrics("timebetweentokens")
            self.plot_metrics("timebetweentokens_median")
            self.plot_metrics("timebetweentokens_p95")

        # Store all data points in DynamoDB
        self.store_data_points()
