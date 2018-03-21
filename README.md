# bdetect-py

## Overview

This repo contains source code for a machine learning system to identify "bullying traces," or indicators of real-life bullying on social media. More info about bullying traces can be found here: http://pages.cs.wisc.edu/~jerryzhu/pub/naaclhlt2012.pdf

The system here improves upon the classification accuracy of the models presented in the above paper. Whereas the UW-Madison paper uses standard n-gram feature space representations / ML models to classify tweets, we use various tree kernel-based SVMs, which have experimentally shown greater accuracy than the above techniques (i.e. in http://www.cs.columbia.edu/~julia/papers/Agarwaletal11.pdf) because they implicitly operate on a richer feature space.

To parse each tweet into a dependency tree so it can be fed to a tree kernel, we use [TweeboParser](https://github.com/ikekonglp/TweeboParser) provided by Kong et al.

## Building

**Supported OSes:** Mac/Linux (only tested on Linux so far)

**Prequisites:**

- Python 3
- Standard build tools (i.e. make)
- GNU Autoconf / m4
- CMake
- Automake (*)
- Perl (*)
- Python 2 (for TweeboParser)
- Java 6 or greater (for POS tagger)

(*) You might not really need this. Haven't tested.

On Ubuntu you can install all of the prequisites with the following commands. Modify as appropriate if you're using a different OS.

```sh
sudo apt-get install python3 build-essential autoconf m4 cmake automake perl python

# Install Java 7 or 8 (depending on your system): https://stackoverflow.com/a/16263651/4077294
# For Ubuntu 16.04 and higher:
sudo apt-get install openjdk-8-jdk
# For other Ubuntu versions:
sudo apt-get install openjdk-7-jdk
```

Once you've taken care of the steps above, run these commands:

```sh
git submodule update --init --recursive
# Type 'n' when prompted whether you have already downloaded models.
python3 main.py
```
