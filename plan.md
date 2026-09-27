

# Update to llama-perplexity
Currently there is no indication of progress when running llama-perplexity. It would be nice to have a progress reported. And also precalculate the number of model execution calls that will happen before running the testing, this will help me figure out how long the test will take and also help me figure out how many calls to the accelerator will happen. This will help me figure out if I can run the test in a reasonable time or not. 




#  BFPP_Acc v4
I want to create BFPPv4 which will include a simple softmax module inside the accelerator/driver/backend. 
I have already created a standalone working version of this accelerator with simpler integration testing and driver in "example_acc" folder. I want to now create bfpp_acc v4 which takes takes the standalone accelerator and integrates it into the SECDA backend and driver, and overall SECDA-LLM framework. This might mean updating ggml-secda to support the softmax offloading. We probable will need to add "SECDA_BFPP_ACC_V4" for driver and backend to support the softmax offloading. 
Do all the testing in simulation with llama-bench and test-backed-ops.





I want to look into llama.cpp implementation of softmax for f32 so that we can add simple softmax module inside the accelerator/driver/backend. Lets plan this and make it into BFFPv4
