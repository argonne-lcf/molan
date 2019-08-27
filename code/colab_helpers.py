import sys
import subprocess
GIT_URL = ''


def is_running_colab():
    return 'google.colab' in sys.modules


def run_cmd(cmd):
    print('Output of "{}":'.format(cmd))
    print(subprocess.run(cmd, stdout=subprocess.PIPE,
                         shell=True).stdout.decode('utf-8'))


def pip_install(package_list):
    run_cmd_list(['pip install {}'.format(p) for p in package_list])


def install_rdkit():
    cmd_list = [
        "wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh",
        "chmod +x Miniconda3-latest-Linux-x86_64.sh",
        "bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local",
        "conda install -q -y -c conda-forge rdkit",
        "rm -rf Miniconda3-latest-Linux-x86_64.sh"]
    print('Installing rdkit\n\n')
    run_cmd_list(cmd_list)
    custom_path = '/usr/local/lib/python3.7/site-packages/'
    print('Do not forget to append "{}"" to sys.path'.format(custom_path))


def run_cmd_list(cmd_list):
    for cmd in cmd_list:
        run_cmd(cmd)
