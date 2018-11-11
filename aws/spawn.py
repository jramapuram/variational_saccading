import os
import pprint
import time
import boto3
import argparse
import paramiko
import numpy as np


parser = argparse.ArgumentParser(description='AWS Spawner')

# Node config
parser.add_argument('--instance-type', type=str, default="p3.2xlarge",
                    help="instance type (default: p3.2xlarge)")
parser.add_argument('--number-of-instances', type=int, default=1,
                    help="number of instances to spawn (default: 1)")
parser.add_argument('--storage-size', type=int, default=150,
                    help="storage in GiB (default: 150)")
parser.add_argument('--keypair', type=str, default="aws",
                    help="keypair name (default: aws)")
parser.add_argument('--security-group', type=str, default="default",
                    help="security group (default: default)")
parser.add_argument('--ami', type=str, default="ami-047daf3f2b162fc35",
                    help="instance AMI (default: ami-047daf3f2b162fc35)")
parser.add_argument('--instance-region', type=str, default="us-east-1",
                    help="instance region (default: us-east-1)")
parser.add_argument('--s3-iam', type=str, default='s3-access',
                    help="iam role for s3 (default: s3-iam)")
parser.add_argument('--upper-bound-spot-multiplier', type=float, default=0,
                    help="sets the upper bound for a spot instance price as a fraction \
                    from the cheapest spot price, eg: 1.3; 0 disables and requests on-demand (default 0)")

# Command config
parser.add_argument('--no-terminate', action='store_true', default=False,
                    help='do not terminate instance after running command (default=False)')
parser.add_argument('--no-background', action='store_true', default=False,
                    help='do not run command in background (default=False)')
parser.add_argument('--cmd', type=str, default=None,
                    help="[required] run this command in the instance (default: None)")
args = parser.parse_args()


def get_all_availability_zones(client):
    '''helper to list all available zones in a region'''
    all_zones_list = client.describe_availability_zones()['AvailabilityZones']
    return [zone['ZoneName'] for zone in all_zones_list]

def get_cheapest_price(args):
    ''' gets the cheapest current AWS spot price, returns price and zone'''
    client = boto3.client('ec2', region_name=args.instance_region)
    price = client.describe_spot_price_history(
        InstanceTypes=[args.instance_type],
        MaxResults=1,
        ProductDescriptions=['Linux/UNIX (Amazon VPC)']# ,
    )
    cheapest_price = float(price['SpotPriceHistory'][0]['SpotPrice'])
    cheapest_zone = price['SpotPriceHistory'][0]['AvailabilityZone']

    # sanity checks
    assert cheapest_price is not None and cheapest_price < np.inf, "cheapest price was not determined"
    assert cheapest_zone is not None and cheapest_zone != "", "could not find cheapest zone"

    # return price and zone
    print("found cheapest price of {}$ in zone {}".format(cheapest_price, cheapest_zone))
    return cheapest_price, cheapest_zone

def instances_to_ips(instance_list):
    assert isinstance(instance_list, list), "need a list as input"

    # create ec2 client
    client = boto3.client('ec2', region_name=args.instance_region)

    # check if they have all spun-up; give it ~2 min
    instances_running = np.array([False for _ in range(len(instance_list))])
    total_waited_time, wait_interval = 0, 5
    max_time_to_wait = 120 / wait_interval # 2 min = 120s / [5s interval] = 24 counts

    # simple loop to wait for instances to get created
    print("waiting for instance spinup.", end='', flush=True)
    while not instances_running.all():
        reservations = client.describe_instances(InstanceIds=instance_list)['Reservations'][0]
        for i, instance in enumerate(reservations['Instances']):
            if instance['State']['Name'] == 'running':
                instances_running[i] = True

            if total_waited_time > max_time_to_wait :
                raise Exception("max time expired for instance creation".format(max_time_to_wait))

            print(".", end='', flush=True)
            time.sleep(wait_interval)
            total_waited_time += 1

    # XXX: add an extra sleep here to get ssh up
    time.sleep(wait_interval*5)
    print("{} instances successfully in running state.".format(len(instance_list)))

    reservations = client.describe_instances(InstanceIds=instance_list)['Reservations'][0]
    return [instance['PublicIpAddress'] for instance in reservations['Instances']]


def create_spot(args):
    try:
        client = boto3.client('ec2', region_name=args.instance_region)
        cheapest_price, cheapest_zone = get_cheapest_price(args)
        max_price = cheapest_price * args.upper_bound_spot_multiplier
        print("setting max price to {}".format(max_price))

        # custom creation of storage dict
        storage_dict = {
            'DeviceName': '/dev/sda1',
            'Ebs':{
                'DeleteOnTermination': True,
                'VolumeType': 'gp2',
                'VolumeSize': args.storage_size
            }
        }

        # create the launch spec dict
        launch_spec_dict = {
            'SecurityGroups': [args.security_group],
            'ImageId': args.ami,
            'KeyName': args.keypair,
            'InstanceType': args.instance_type,
            'BlockDeviceMappings': [storage_dict],
            'IamInstanceProfile': {
                'Name': args.s3_iam
            }
        }

        # request the node(s)
        instance_request = client.request_spot_instances(
            SpotPrice=str(max_price),
            Type='one-time',
            InstanceInterruptionBehavior='terminate',
            InstanceCount=args.number_of_instances,
            LaunchSpecification=launch_spec_dict
        )
        #print("\ninstance_request = {}\n".format(instance_request))

        # return the requested instances
        instance_request_id = instance_request['SpotInstanceRequests'][0]['SpotInstanceRequestId']
        print("spot-request-id: ", instance_request_id)

        # wait till full-fulled, upto 10 min
        instances, all_instances_created = [], False
        total_waited_time, wait_interval = 0, 5
        max_time_to_wait = 600 / wait_interval # 10 min = 600s / [5s interval] = 120 counts
        print("creating {} instances, please be patient.".format(args.number_of_instances), end='', flush=True)

        while not all_instances_created:
            spot_req_response = client.describe_spot_instance_requests(
                SpotInstanceRequestIds=[instance_request_id]
            )
            #print("\nspot_req = {}\n".format(spot_req_response))
            for spot_req in spot_req_response['SpotInstanceRequests']:
                if spot_req['State'] == "failed":
                    raise Exception("spot request failed:", spot_req)

                print(".", end='', flush=True)
                if 'InstanceId' not in spot_req or not spot_req['InstanceId']:
                    if total_waited_time > max_time_to_wait :
                        raise Exception("max time expired; instance creation failed".format(max_time_to_wait))

                    # sleep and increment wait count
                    time.sleep(wait_interval)
                    total_waited_time += 1
                    break

                # add the instance to the list of created instances
                instance_id = spot_req['InstanceId']
                instances.append(instance_id)
                if len(instances) == args.number_of_instances:
                    print("successfully created {} instances".format(len(instances)))
                    all_instances_created = True

        return instances_to_ips(instances)

    except BaseException as exe:
        print(exe)


def create_ondemand(args):
    raise NotImplementedError("on-demand not-yet implemented")
    try:
        ec2 = boto3.resource('ec2', region_name=args.instance_zone)
        subnet = ec2.Subnet(args.subnet) if args.subnet is not None else ec2.Subnet()
        instances = subnet.create_instances(
            ImageId=args.ami,
            InstanceType=args.instance_type,
            MaxCount=args.number_of_instances,
            MinCount=args.number_of_instances,
            KeyName=args.keypair,
            SecurityGroups=[args.security_group]
        )

        # return the requested instances
        print(instances)
        return instances

    except BaseException as exe:
        print(exe)


def run_command(cmd, hostname, pem_file, username='ubuntu',
                background=True, terminate_on_completion=True):
    key = paramiko.RSAKey.from_private_key_file(pem_file)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # format the command
    shutdown_cmd = "&& sudo shutdown -h 0" if terminate_on_completion else ""
    if background:
        cmd = "nohup sh -c \"{} {}\" &".format(
            cmd, shutdown_cmd
        )
    else:
        cmd = "{} {}".format(
            cmd, shutdown_cmd
        )

    # try to spin-up the command on the remote instance
    client.connect(hostname=hostname, username=username, pkey=key)
    stdin, stdout, stderr = client.exec_command(cmd)
    stdout = stdout.read().decode('ascii')
    stderr = stderr.read().decode('ascii')
    client.close()

    # return the text in string format
    return {
        'stdout': stdout,
        'stderr': stderr
    }


if __name__ == "__main__":
    # print the config and sanity check
    pprint.PrettyPrinter(indent=4).pprint(vars(args))
    assert args.cmd is not None, "need to specify a command"

    # execute the config
    if args.upper_bound_spot_multiplier > 0:
        print("attempted to request spot instance w/max price = {} x cheapest".format(
            args.upper_bound_spot_multiplier)
        )
        instance_ips = create_spot(args)
    else:
        print("creating on-demand instance")
        instance_ips = create_ondemand(args)

    # run the specified command over ssh
    # also append shutdown command to terminate
    pem_file_path = os.path.join(os.path.expanduser('~'), ".ssh", args.keypair + ".pem")
    for instance_ip in instance_ips:
        cli_out = run_command(args.cmd, instance_ip, pem_file_path,
                              background=not args.no_background,
                              terminate_on_completion=not args.no_terminate)
        print("[{}][stdout]: {}".format(instance_ip, cli_out['stdout']))
        print("[{}][stderr]: {}".format(instance_ip, cli_out['stderr']))
