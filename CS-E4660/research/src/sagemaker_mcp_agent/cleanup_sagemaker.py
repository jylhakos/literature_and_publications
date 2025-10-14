"""
Cleanup AWS SageMaker Resources

This script deletes SageMaker endpoints, endpoint configurations,
and models to avoid unnecessary charges.

Usage:
    python cleanup_sagemaker.py --endpoint Arcee-Agent-2025-01-13-10-30-00
    python cleanup_sagemaker.py --all  # Delete all Arcee-Agent endpoints
"""

import argparse
import boto3
import sagemaker


def delete_endpoint(endpoint_name: str, delete_config: bool = True, delete_model: bool = True):
    """
    Delete a SageMaker endpoint and optionally its configuration and model.
    
    Args:
        endpoint_name: Name of the endpoint to delete
        delete_config: Whether to delete endpoint configuration
        delete_model: Whether to delete the model
    """
    client = boto3.client('sagemaker')
    session = sagemaker.Session()
    
    print(f"\n🗑️  Deleting endpoint: {endpoint_name}")
    
    try:
        # Get endpoint details
        endpoint_desc = client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = endpoint_desc['EndpointConfigName']
        
        # Delete endpoint
        print(f"   Deleting endpoint...")
        session.delete_endpoint(endpoint_name)
        print(f"   ✓ Endpoint deleted")
        
        if delete_config:
            # Delete endpoint configuration
            print(f"   Deleting endpoint configuration...")
            session.delete_endpoint_config(endpoint_config_name)
            print(f"   ✓ Endpoint configuration deleted")
            
            if delete_model:
                try:
                    # Get model name from endpoint config
                    config_desc = client.describe_endpoint_config(
                        EndpointConfigName=endpoint_config_name
                    )
                    
                    # Delete models
                    for variant in config_desc['ProductionVariants']:
                        model_name = variant['ModelName']
                        print(f"   Deleting model: {model_name}...")
                        
                        try:
                            client.delete_model(ModelName=model_name)
                            print(f"   ✓ Model deleted: {model_name}")
                        except client.exceptions.ClientError as e:
                            if 'does not exist' in str(e):
                                print(f"   ⚠️  Model already deleted: {model_name}")
                            else:
                                raise
                                
                except Exception as e:
                    print(f"   ⚠️  Could not delete model: {e}")
        
        print(f"✅ Cleanup complete for {endpoint_name}\n")
        
    except client.exceptions.ClientError as e:
        if 'Could not find endpoint' in str(e):
            print(f"   ⚠️  Endpoint not found: {endpoint_name}")
        else:
            print(f"   ❌ Error: {e}")
            raise


def list_endpoints(prefix: str = "Arcee-Agent"):
    """
    List all SageMaker endpoints with the given prefix.
    
    Args:
        prefix: Endpoint name prefix to filter
        
    Returns:
        List of endpoint names
    """
    client = boto3.client('sagemaker')
    
    paginator = client.get_paginator('list_endpoints')
    endpoints = []
    
    for page in paginator.paginate():
        for endpoint in page['Endpoints']:
            if endpoint['EndpointName'].startswith(prefix):
                endpoints.append({
                    'name': endpoint['EndpointName'],
                    'status': endpoint['EndpointStatus'],
                    'created': endpoint['CreationTime']
                })
    
    return endpoints


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cleanup AWS SageMaker resources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Delete a specific endpoint
  python cleanup_sagemaker.py --endpoint Arcee-Agent-2025-01-13-10-30-00
  
  # List all Arcee-Agent endpoints
  python cleanup_sagemaker.py --list
  
  # Delete all Arcee-Agent endpoints
  python cleanup_sagemaker.py --all
        """
    )
    
    parser.add_argument(
        "--endpoint",
        help="Name of the endpoint to delete"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all Arcee-Agent endpoints"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete all Arcee-Agent endpoints"
    )
    
    parser.add_argument(
        "--prefix",
        default="Arcee-Agent",
        help="Endpoint name prefix (default: Arcee-Agent)"
    )
    
    parser.add_argument(
        "--keep-config",
        action="store_true",
        help="Keep endpoint configuration"
    )
    
    parser.add_argument(
        "--keep-model",
        action="store_true",
        help="Keep model"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.endpoint, args.list, args.all]):
        parser.error("Specify --endpoint, --list, or --all")
    
    # List endpoints
    if args.list or args.all:
        print("\n" + "=" * 70)
        print("SageMaker Endpoints")
        print("=" * 70)
        
        endpoints = list_endpoints(args.prefix)
        
        if not endpoints:
            print(f"No endpoints found with prefix '{args.prefix}'")
        else:
            print(f"\nFound {len(endpoints)} endpoint(s):\n")
            for ep in endpoints:
                print(f"  • {ep['name']}")
                print(f"    Status: {ep['status']}")
                print(f"    Created: {ep['created']}")
                print()
        
        if args.list:
            return 0
    
    # Delete specific endpoint
    if args.endpoint:
        try:
            delete_endpoint(
                args.endpoint,
                delete_config=not args.keep_config,
                delete_model=not args.keep_model
            )
            return 0
        except Exception as e:
            print(f"\n❌ Error deleting endpoint: {e}")
            return 1
    
    # Delete all endpoints
    if args.all:
        endpoints = list_endpoints(args.prefix)
        
        if not endpoints:
            print(f"\nNo endpoints to delete")
            return 0
        
        print(f"\n⚠️  About to delete {len(endpoints)} endpoint(s)")
        print("   This action cannot be undone!")
        
        confirm = input("\nType 'yes' to confirm: ").strip().lower()
        
        if confirm != 'yes':
            print("❌ Cancelled")
            return 1
        
        print()
        success_count = 0
        error_count = 0
        
        for ep in endpoints:
            try:
                delete_endpoint(
                    ep['name'],
                    delete_config=not args.keep_config,
                    delete_model=not args.keep_model
                )
                success_count += 1
            except Exception as e:
                print(f"❌ Error deleting {ep['name']}: {e}\n")
                error_count += 1
        
        print("=" * 70)
        print(f"✅ Deleted {success_count} endpoint(s)")
        if error_count > 0:
            print(f"❌ Failed to delete {error_count} endpoint(s)")
        print("=" * 70)
        
        return 0 if error_count == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
