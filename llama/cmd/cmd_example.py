import subprocess
import os
ps1_value = os.environ.get("PS1")

while True:
    p = "~"
    cmd = input("(why):" + p + '$')
    b = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout
    print("(why):" + p + '$' + b.decode('gbk'))