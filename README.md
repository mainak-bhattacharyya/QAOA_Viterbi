# QAOA_Viterbi

Code repository authored by [Mainak Bhattacharyya](https://mainak-bhattacharyya.github.io/) for the research article titled 
<center>

**Quantum Approximation Optimization Algorithm for the trellis based Viterbi decoding of classical error correcting codes**

</center>

Listed at arxiv [2304.02292](https://arxiv.org/abs/2304.02292) under quant-ph

Dev setup
-----------

Please follow the guidelines below to setup the python environment with necessary dependencies on a Linux machine.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Explorer guide
---------------

- The cost landscape plots can be generated using the scripts from the directory `./cost_landscape_scripts`. Some sample data generated through simulation of the scripts are available at directory `./data` under the preffix *land*.
- Scripts for **Word Error Rate** estimation can be found at the directory `error_rate_scripts`.
- For any issue with the scripts please drop a mail at googol.mainak@gmail.com

Copyright
----------

Copyright (c) [2025] [Mainak Bhattacharyya]

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. This work is copyrighted. Do not incorporate it into any commercial product without prior written permission from the copyright holder.

2. Any publications or derivative works based on this work must cite the paper by Bhattacharyya and Raina.

3. Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.

4. Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.

5. The name of the copyright holder or contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

-----------------------------------