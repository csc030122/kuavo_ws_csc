cp hipnuc_imu_serial.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo service udev reload
sudo service udev restart
