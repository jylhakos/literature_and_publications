"""
Deploy Arcee Agent to AWS SageMaker

This script automates the deployment of Arcee Agent to AWS SageMaker,
creating a scalable, managed endpoint for inference.

Usage:
    python deploy_sagemaker.py --instance-type ml.g5.2xlarge
    python deploy_sagemaker.py --instance-type ml.g5.12xlarge --gpus 4

Requirements:
    - AWS credentials configured
    - Sufficient SageMaker quota for chosen instance type
    - boto3 and sagemaker packages installed
"""

import argparse
import datetime
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri


def deploy_arcee_agent(
    model_id: str = "arcee-ai/Arcee-Agent",
    instance_type: str = "ml.g5.2xlarge",
    num_gpus: int = 1,
    endpoint_name_prefix: str = "Arcee-Agent",
    enable_messages_api: bool = True,
    max_input_tokens: int = None,
    max_total_tokens: int = None
):
    """
    Deploy Arcee Agent to AWS SageMaker.
    
    Args:
        model_id: Hugging Face model ID
        instance_type: SageMaker instance type
        num_gpus: Number of GPUs to use
        endpoint_name_prefix: Prefix for endpoint name
        enable_messages_api: Enable OpenAI-compatible API
        max_input_tokens: Maximum input tokens (optional)
        max_total_tokens: Maximum total tokens (optional)
    
    Returns:
        Endpoint name
    """
    
    print("=" * 70)
    print("Deploying Arcee Agent to AWS SageMaker")
    print("=" * 70)
    print(f"Model ID: {model_id}")
    print(f"Instance Type: {instance_type}")
    print(f"Number of GPUs: {num_gpus}")
    print()
    
    # Get AWS configuration
    role = get_execution_role()
    sagemaker_session = sagemaker.Session()
    region = boto3.Session().region_name
    
    print(f"AWS Region: {region}")
    print(f"SageMaker Execution Role: {role}")
    print()
    
    # Configure model environment
    model_environment = {
        "HF_MODEL_ID": model_id,
        "SM_NUM_GPUS": str(num_gpus),
    }
    
    if enable_messages_api:
        model_environment["MESSAGES_API_ENABLED"] = "true"
        print("✓ OpenAI-compatible Messages API enabled")
    
    if max_input_tokens:
        model_environment["MAX_INPUT_TOKENS"] = str(max_input_tokens)
        print(f"✓ Max input tokens: {max_input_tokens}")
    
    if max_total_tokens:
        model_environment["MAX_TOTAL_TOKENS"] = str(max_total_tokens)
        print(f"✓ Max total tokens: {max_total_tokens}")
    
    print()
    
    # Create deployable model
    print("Creating HuggingFace model...")
    model = HuggingFaceModel(
        role=role,
        env=model_environment,
        image_uri=get_huggingface_llm_image_uri("huggingface", version="2.2.0"),
    )
    print("✓ Model created")
    
    # Generate unique endpoint name
    timestamp = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now())
    endpoint_name = f"{endpoint_name_prefix}-{timestamp}"
    
    print(f"\nDeploying endpoint: {endpoint_name}")
    print("This may take 5-10 minutes...")
    print()
    
    # Deploy to endpoint
    try:
        response = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            model_data_download_timeout=3600,
            container_startup_health_check_timeout=900,
        )
        
        print()
        print("=" * 70)
        print("✅ Deployment Successful!")
        print("=" * 70)
        print(f"Endpoint Name: {endpoint_name}")
        print(f"Endpoint ARN: {response.endpoint_arn}")
        print()
        print("💡 To use this endpoint, set the environment variable:")
        print(f"   export SAGEMAKER_ENDPOINT={endpoint_name}")
        print()
        print("📊 Estimated costs:")
        if instance_type == "ml.g5.2xlarge":
            print("   ~$1.41/hour (~$1,029/month if running 24/7)")
        elif instance_type == "ml.g5.12xlarge":
            print("   ~$7.09/hour (~$5,176/month if running 24/7)")
        print()
        print("⚠️  Remember to delete the endpoint when not in use:")
        print(f"   python cleanup_sagemaker.py --endpoint {endpoint_name}")
        print("=" * 70)
        
        return endpoint_name
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ Deployment Failed!")
        print("=" * 70)
        print(f"Error: {str(e)}")
        print()
        print("Common issues:")
        print("  1. Insufficient quota for instance type")
        print("     Solution: Request quota increase in AWS Service Quotas")
        print("  2. Invalid IAM permissions")
        print("     Solution: Ensure role has SageMaker permissions")
        print("  3. Instance type not available in region")
        print("     Solution: Choose different instance type or region")
        print("=" * 70)
        raise


def check_prerequisites():
    """Check if prerequisites are met."""
    print("\nChecking prerequisites...")
    
    # Check AWS credentials
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        if not credentials:
            raise ValueError("No AWS credentials found")
        print("✓ AWS credentials configured")
    except Exception as e:
        print(f"❌ AWS credentials not found: {e}")
        print("\nPlease configure AWS credentials:")
        print("  aws configure")
        print("  OR set environment variables:")
        print("    export AWS_ACCESS_KEY_ID=...")
        print("    export AWS_SECRET_ACCESS_KEY=...")
        print("    export AWS_DEFAULT_REGION=...")
        return False
    
    # Check SageMaker permissions
    try:
        client = boto3.client('sagemaker')
        client.list_endpoints(MaxResults=1)
        print("✓ SageMaker access verified")
    except Exception as e:
        print(f"❌ Cannot access SageMaker: {e}")
        print("\nEnsure your IAM role has SageMaker permissions")
        return False
    
    # Check boto3/sagemaker packages
    try:
        import sagemaker
        print(f"✓ SageMaker SDK version: {sagemaker.__version__}")
    except ImportError:
        print("❌ SageMaker SDK not installed")
        print("\nInstall with: pip install sagemaker")
        return False
    
    print()
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy Arcee Agent to AWS SageMaker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy with default settings (ml.g5.2xlarge, 1 GPU)
  python deploy_sagemaker.py
  
  # Deploy with larger instance (ml.g5.12xlarge, 4 GPUs)
  python deploy_sagemaker.py --instance-type ml.g5.12xlarge --gpus 4
  
  # Deploy with custom token limits
  python deploy_sagemaker.py --max-input-tokens 16384 --max-total-tokens 32768
        """
    )
    
    parser.add_argument(
        "--model-id",
        default="arcee-ai/Arcee-Agent",
        help="Hugging Face model ID (default: arcee-ai/Arcee-Agent)"
    )
    
    parser.add_argument(
        "--instance-type",
        default="ml.g5.2xlarge",
        choices=["ml.g5.2xlarge", "ml.g5.12xlarge", "ml.g5.xlarge"],
        help="SageMaker instance type (default: ml.g5.2xlarge)"
    )
    
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)"
    )
    
    parser.add_argument(
        "--endpoint-name-prefix",
        default="Arcee-Agent",
        help="Prefix for endpoint name (default: Arcee-Agent)"
    )
    
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        help="Maximum input tokens (optional)"
    )
    
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        help="Maximum total tokens (optional)"
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip prerequisite checks"
    )
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            print("\n❌ Prerequisites not met. Exiting.")
            return 1
    
    # Deploy
    try:
        endpoint_name = deploy_arcee_agent(
            model_id=args.model_id,
            instance_type=args.instance_type,
            num_gpus=args.gpus,
            endpoint_name_prefix=args.endpoint_name_prefix,
            enable_messages_api=True,
            max_input_tokens=args.max_input_tokens,
            max_total_tokens=args.max_total_tokens
        )
        
        # Save endpoint name to file
        with open(".sagemaker_endpoint", "w") as f:
            f.write(endpoint_name)
        print(f"✓ Endpoint name saved to .sagemaker_endpoint")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
