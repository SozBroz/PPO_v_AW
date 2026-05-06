@echo off
ssh sshuser@192.168.0.160 "hostname && cd D:\awbw && dir checkpoints\value_rhea_latest.pt 2>&1"
