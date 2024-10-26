from motor_delay_fft import MotorDelay_130
import torch
motordelay2 = MotorDelay_130(4096, 1,device="cuda:0")

tau_ = torch.ones(4096, dtype=torch.float,requires_grad=False,device="cuda:0")
tau_ = tau_ * 500
tau = motordelay2(tau_)
print(tau)