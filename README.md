# AimBot
An AI-powered aimbot written in Rust

## Share devices from Host to WSL
Open **Windows PowerShell** as administrator.

Install **usbipd** if it's not already installed by running ```winget install usbipd-win```
1. List all device
```shell
usbipd list
```
```text
# Sample output:
Connected:
BUSID  VID:PID    DEVICE                                                        STATE
1-1    046d:c548  Logitech USB Input Device, USB Input Device                   Not shared
1-2    1a86:55d3  USB-Enhanced-SERIAL CH343 (COM3)                              Not shared
1-5    8087:0032  Intel(R) Wireless Bluetooth(R)                                Not shared
```
2. Bind the usb to share
```shell
usbipd bind --busid 1-2
```
3. Verify its shared
```shell
usbipd list
```
```text
# Sample output:
Connected:
BUSID  VID:PID    DEVICE                                                        STATE
1-1    046d:c548  Logitech USB Input Device, USB Input Device                   Not shared
1-2    1a86:55d3  USB-Enhanced-SERIAL CH343 (COM3)                              Shared
1-5    8087:0032  Intel(R) Wireless Bluetooth(R)                                Not shared
```
4. Attach to WSL
```shell
usbipd attach --wsl --busid 1-2
```
```text
# Sample output:
usbipd: info: Using WSL distribution 'Ubuntu-24.04' to attach; the device will be available in all WSL 2 distributions.
usbipd: info: Loading vhci_hcd module.
usbipd: info: Detected networking mode 'nat'.
usbipd: info: Using IP address 172.27.32.1 to reach the host.
```
5. Check port in WSL
```shell
espflash board-info
```
```text
# Sample output:
[2025-08-17T05:44:28Z INFO ] Serial port: '/dev/ttyACM0'
[2025-08-17T05:44:28Z INFO ] Connecting...
```


## Prepare a YOLO object detection model. 
The model is defined with two classes:
- 0: Entire body
- 1: Head


## Setup event listener

To forward a port from a WSL instance to your Windows host, thereby making a service running in WSL accessible from Windows or even other devices on your local network, you can use the netsh interface portproxy command in Windows PowerShell.

Open your WSL terminal and find the IP address of your WSL instance. Common commands are ip addr show or hostname -I. Note down the IP address (it will likely be in the 172.x.x.x range). Configure Port Forwarding in PowerShell.

Open Windows PowerShell as an administrator. Execute the following command, replacing [LISTEN_PORT] with the port you want to access on Windows, [WSL_IP] with the IP address of your WSL instance, and [WSL_PORT] with the port the service is listening on inside WSL:

```shell
netsh interface portproxy add v4tov4 listenport=[LISTEN_PORT] listenaddress=0.0.0.0 connectport=[WSL_PORT] connectaddress=[WSL_IP]
```

Open Windows Firewall (if necessary). If the Windows Firewall is blocking inbound connections to the [LISTEN_PORT], you will need to create an inbound rule. In PowerShell (as administrator), run:

```shell
New-NetFirewallRule -DisplayName "Allow TCP on Port [LISTEN_PORT]" -Direction Inbound -Action Allow -Protocol TCP -LocalPort [LISTEN_PORT]
```

To list all the **inbound** rule:

```shell
Get-NetFirewallRule -Direction Inbound
```

To delete a rule:

```shell
netsh advfirewall firewall delete rule name="Rule Name"
```

Find your Windows [IP] address by

```shell
ipconfig
```

From another device on the same network, access the board controller at: [http://[IP]:[LISTEN_PORT]/stream/board]()