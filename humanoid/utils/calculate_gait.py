# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def get_coefficients(h0, hswing, v0, vswing, hmax, swing_time, acc):
    def equations(coeffs):
        a5, a4, a3, a2, a1, a0 = coeffs
        # Height at t=0 should be h0
        eq1 = a0 - h0

        # Height at t=swing_time should be hswing
        eq2 = a5 * swing_time ** 5 + a4 * swing_time ** 4 + a3 * swing_time ** 3 + a2 * swing_time ** 2 + a1 * swing_time + a0 - hswing

        # Velocity at t=0 should be v0
        eq3 = a1 - v0

        # Velocity at t=swing_time should be vswing
        eq4 = 5 * a5 * swing_time ** 4 + 4 * a4 * swing_time ** 3 + 3 * a3 * swing_time ** 2 + 2 * a2 * swing_time + a1 - vswing

        # Height at t=swing_time/2 should be hmax
        eq5 = a5 * (swing_time / 2.0) ** 5 + a4 * (swing_time / 2.0) ** 4 + a3 * (swing_time / 2.0) ** 3 + a2 * (
                    swing_time / 2.0) ** 2 + a1 * (swing_time / 2.0) + a0 - hmax

        eq6 = 2 * a2 - acc
        # Return the deviations from the expected values. These will be minimized by fsolve.
        return (eq1, eq2, eq3, eq4, eq5, eq6)  # , a5 + a4 + a3 + a2 + a1 + a0)

    # Solve for the coefficients using the equations above
    return fsolve(equations, (1, 1, 1, 1, 1, 1))


def plot_curves(coeffs, swing_time):
    a5, a4, a3, a2, a1, a0 = coeffs

    def h(t):
        return a5 * t ** 5 + a4 * t ** 4 + a3 * t ** 3 + a2 * t ** 2 + a1 * t + a0

    def v(t):
        return 5 * a5 * t ** 4 + 4 * a4 * t ** 3 + 3 * a3 * t ** 2 + 2 * a2 * t + a1

    # Define the acceleration function based on the coefficients
    def a(t):
        return 20 * a5 * t ** 3 + 12 * a4 * t ** 2 + 6 * a3 * t + 2 * a2

    t_values = np.linspace(0, swing_time, 500)
    h_values = h(t_values)
    v_values = v(t_values)
    a_values = a(t_values)  # Compute acceleration values

    discrete_t_values = np.linspace(0, swing_time, 14)
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(12, 8))

    # plt.subplot(3, 1, 1)
    # plt.plot(t_values, h_values, label='Height (h(t))')
    # plt.scatter(discrete_t_values, h(discrete_t_values), color='black', label='Discrete Height')
    # plt.title('Height Curve')
    # plt.grid(True)
    # plt.legend()
    #
    # plt.subplot(3, 1, 2)
    # plt.plot(t_values, v_values, label='Velocity (v(t))', color='red')
    # plt.scatter(discrete_t_values, v(discrete_t_values), color='black', label='Discrete Velocity')
    # plt.title('Velocity Curve')
    # plt.grid(True)
    # plt.legend()
    #
    # # Plotting the acceleration curve
    # plt.subplot(3, 1, 3)
    # plt.plot(t_values, a_values/50, label='Acceleration (a(t))', color='green')
    # plt.scatter(discrete_t_values, a(discrete_t_values)/50, color='black', label='Discrete Acceleration')
    # plt.title('Acceleration Curve')
    # plt.grid(True)
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()
    plt.subplot(2, 1, 1)
    plt.plot(t_values, h_values, linewidth=3, label='Height (h(t))', color='black')
    # plt.scatter(discrete_t_values, h(discrete_t_values), color='black')
    plt.title('Height Curve', fontsize=30)
    plt.grid(True, linewidth=3)  # 增加网格线的粗细
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=22)

    plt.subplot(2, 1, 2)
    plt.plot(t_values, v_values, linewidth=3, label='Velocity (v(t))', color='blue')
    # plt.scatter(discrete_t_values, v(discrete_t_values), color='black')
    plt.title('Velocity Curve', fontsize=30)
    plt.grid(True, linewidth=3)  # 增加网格线的粗细
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=22)

    plt.tight_layout()
    # plt.show()
    plt.savefig('curves.png')


# # Set the constraints and swing time
# coeffs = get_coefficients(0, 0, 0.1, 0.0, 0.05, 0.32,10)
coeffs = get_coefficients(0, 0, 0.1, 0.0, 0.1, 0.32, 25)

# print(f"a5 = {coeffs[0]:.15f}")
# print(f"a4 = {coeffs[1]:.15f}")
# print(f"a3 = {coeffs[2]:.15f}")
# print(f"a2 = {coeffs[3]:.15f}")
# print(f"a1 = {coeffs[4]:.15f}")
# print(f"a0 = {coeffs[5]:.15f}")
# coeffs = [9.6,12,-18.8,5.0,0.1,0.0]
# print("coeffs",coeffs)
# print("Coefficients (a5, a4, a3, a2, a1, a0):")
# print(f"a5 = {coeffs[0]:.15f}")
# print(f"a4 = {coeffs[1]:.15f}")
# print(f"a3 = {coeffs[2]:.15f}")
# print(f"a2 = {coeffs[3]:.15f}")
# print(f"a1 = {coeffs[4]:.15f}")
# print(f"a0 = {coeffs[5]:.15f}")
# #
# plot_curves(coeffs, 0.32)