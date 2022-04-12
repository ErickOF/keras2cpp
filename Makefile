.PONY: run build clean

CC=g++
SRC=test_run_cnn.cc keras_model.cc
OBJS=$(patsubst %.cpp,%.o,$(SRC))
FLAGS=-std=c++11 -Wall


# Run build CNN
run: build
		@./$< $(DUMPED) $(DATA_SAMPLE) $(KERAS2CPP_OUTPUT) \
			|| echo "Run as: run DATA_SAMPLE=path/to/data_sample.dat KERAS2CPP_OUTPUT=keras2cpp_output.dat

# Build CNN
build: $(OBJS)
		@$(CC) $(FLAGS) -o $@ $(OBJS)

# Clean build
clean: build
		@rm $<

# Conver keras model to cpp code
convert-keras2cpp:
		@python dump_to_simple_cpp.py -a $(ARCH) -w $(WEIGHTS) -o $(DUMPED) -v $(VERBOSE) \
			|| echo "Run as: make convert-keras2cpp ARCH=path/to/arch.json WEIGHTS=path/to/weights.h5 DUMPED=dumped.dumped VERBOSE=0|1"

# Generate a random input example
generate-data-sample:
		@python test_run_cnn.py -a $(ARCH) -w $(WEIGHTS) -d $(DATA_SAMPLE) -o $(KERAS_OUTPUT) \
			|| echo "Run as: make generate-data-sample ARCH=path/to/arch.json WEIGHTS=path/to/weights.h5 DATA_SAMPLE=data_sample.dat KERAS_OUTPUT=keras_output.dat"
