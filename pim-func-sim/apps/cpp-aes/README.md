# 0. Environment Specification
```
OS: Ubuntu 16.04 LTS, 64-bit
Memory: 11.5 GiB
CPU: Intel Core i5-7200U @ 2.50GHz × 4
GPU: GeForce 940MX/PCIe/SSE2
```

# 1. AES-ON-GPU
Implement AES on CPU & GPU

# 2. Usage
```
gcc cpu.c -o cpu
```

```
nvcc gpu.cu -o gpu
```

# 3. Experimental Results
<table class="tg">
  <tr>
    <th class="tg-us36" rowspan="3">File Size (bytes)</th>
    <th class="tg-us36" colspan="4">Encryption</th>
    <th class="tg-yw4l" colspan="4">Decryption</th>
  </tr>
  <tr>
    <td class="tg-us36" colspan="2">Time (seconds)</td>
    <td class="tg-yw4l" colspan="2">Throughput (bytes per second)</td>
    <td class="tg-yw4l" colspan="2">Time (seconds)</td>
    <td class="tg-yw4l" colspan="2">Throughput (bytes per second)</td>
  </tr>
  <tr>
    <td class="tg-us36">CPU</td>
    <td class="tg-us36">GPU</td>
    <td class="tg-yw4l">CPU</td>
    <td class="tg-yw4l">GPU</td>
    <td class="tg-yw4l">CPU</td>
    <td class="tg-yw4l">GPU</td>
    <td class="tg-yw4l">CPU</td>
    <td class="tg-yw4l">GPU</td>
  </tr>
  <tr>
    <td class="tg-yw4l">1202</td>
    <td class="tg-yw4l">0.000152</td>
    <td class="tg-yw4l">0.000732</td>
    <td class="tg-yw4l">7921052.63</td>
    <td class="tg-yw4l">1644808.74</td>
    <td class="tg-yw4l">0.000264</td>
    <td class="tg-yw4l">0.000771</td>
    <td class="tg-yw4l">4560606.06</td>
    <td class="tg-yw4l">1561608.30</td>
  </tr>
  <tr>
    <td class="tg-yw4l">4652</td>
    <td class="tg-yw4l">0.000689</td>
    <td class="tg-yw4l">0.000770</td>
    <td class="tg-yw4l">6769230.77</td>
    <td class="tg-yw4l">6057142.85</td>
    <td class="tg-yw4l">0.001168</td>
    <td class="tg-yw4l">0.000821</td>
    <td class="tg-yw4l">3993150.68</td>
    <td class="tg-yw4l">5680876.98</td>
  </tr>
  <tr>
    <td class="tg-yw4l">9302</td>
    <td class="tg-yw4l">0.001418</td>
    <td class="tg-yw4l">0.000784</td>
    <td class="tg-yw4l">6564174.89</td>
    <td class="tg-yw4l">11872448.97</td>
    <td class="tg-yw4l">0.002081</td>
    <td class="tg-yw4l">0.000790</td>
    <td class="tg-yw4l">4472849.59</td>
    <td class="tg-yw4l">11782278.48</td>
  </tr>
  <tr>
    <td class="tg-yw4l">18602</td>
    <td class="tg-yw4l">0.002545</td>
    <td class="tg-yw4l">0.000807</td>
    <td class="tg-yw4l">7313163.06</td>
    <td class="tg-yw4l">23063197.03</td>
    <td class="tg-yw4l">0.003889</td>
    <td class="tg-yw4l">0.000821</td>
    <td class="tg-yw4l">4785806.12</td>
    <td class="tg-yw4l">22669914.74</td>
  </tr>
  <tr>
    <td class="tg-yw4l">37202</td>
    <td class="tg-yw4l">0.004985</td>
    <td class="tg-yw4l">0.000886</td>
    <td class="tg-yw4l">7463189.57</td>
    <td class="tg-yw4l">41990970.65</td>
    <td class="tg-yw4l">0.007196</td>
    <td class="tg-yw4l">0.000918</td>
    <td class="tg-yw4l">5170094.49</td>
    <td class="tg-yw4l">40527233.12</td>
  </tr>
  <tr>
    <td class="tg-yw4l">74402</td>
    <td class="tg-yw4l">0.010099</td>
    <td class="tg-yw4l">0.001048</td>
    <td class="tg-yw4l">7367462.12</td>
    <td class="tg-yw4l">70996183.21</td>
    <td class="tg-yw4l">0.014296</td>
    <td class="tg-yw4l">0.001040</td>
    <td class="tg-yw4l">5204532.74</td>
    <td class="tg-yw4l">71542307.69</td>
  </tr>
  <tr>
    <td class="tg-yw4l">148802</td>
    <td class="tg-yw4l">0.020109</td>
    <td class="tg-yw4l">0.001302</td>
    <td class="tg-yw4l">7399870.70</td>
    <td class="tg-yw4l">114288786.48</td>
    <td class="tg-yw4l">0.028493</td>
    <td class="tg-yw4l">0.001331</td>
    <td class="tg-yw4l">5222475.69</td>
    <td class="tg-yw4l">111798647.63</td>
  </tr>
  <tr>
    <td class="tg-yw4l">297602</td>
    <td class="tg-yw4l">0.039651</td>
    <td class="tg-yw4l">0.001896</td>
    <td class="tg-yw4l">7505586.24</td>
    <td class="tg-yw4l">156964135.02</td>
    <td class="tg-yw4l">0.056839</td>
    <td class="tg-yw4l">0.002018</td>
    <td class="tg-yw4l">5235911.96</td>
    <td class="tg-yw4l">147474727.45</td>
  </tr>
  <tr>
    <td class="tg-yw4l">595202</td>
    <td class="tg-yw4l">0.079503</td>
    <td class="tg-yw4l">0.003145</td>
    <td class="tg-yw4l">7486560.26</td>
    <td class="tg-yw4l">189254054.05</td>
    <td class="tg-yw4l">0.113610</td>
    <td class="tg-yw4l">0.003577</td>
    <td class="tg-yw4l">5239010.65</td>
    <td class="tg-yw4l">166397539.84</td>
  </tr>
  <tr>
    <td class="tg-yw4l">1190402</td>
    <td class="tg-yw4l">0.158507</td>
    <td class="tg-yw4l">0.005316</td>
    <td class="tg-yw4l">7510103.65</td>
    <td class="tg-yw4l">223928517.68</td>
    <td class="tg-yw4l">0.227203</td>
    <td class="tg-yw4l">0.005847</td>
    <td class="tg-yw4l">5239385.04</td>
    <td class="tg-yw4l">203592269.54</td>
  </tr>
</table>
