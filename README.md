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
