# Source - https://stackoverflow.com/a
# Posted by MD BILLAL HOSSAIN, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-14, License - CC BY-SA 4.0

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update

# Source - https://stackoverflow.com/a
# Posted by MD BILLAL HOSSAIN, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-14, License - CC BY-SA 4.0

sudo apt-get install -y nvidia-container-toolkit

# Source - https://stackoverflow.com/a
# Posted by MD BILLAL HOSSAIN, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-14, License - CC BY-SA 4.0

sudo systemctl restart docker
