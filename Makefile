CXX=g++
OUTPUT_SUFFIX=.out
CXXFLAGS=-g -Wall 
LDFLAGS=-lpthread
TARGET=stream

# if both m5ops_header_path and m5_build_path are defined, we build the STREAM b
enchmark with the m5 annotations
ifneq ($(M5_BUILD_PATH),)
	CXXFLAGS += -I$(M5OPS_HEADER_PATH)
	CXXFLAGS += -O -DGEM5_ANNOTATION=1
	LDFLAGS += -lm5 -L$(M5_BUILD_PATH)out/
	OUTPUT_SUFFIX = .m5
endif

all: $(TARGET)$(OUTPUT_SUFFIX)

$(TARGET)$(OUTPUT_SUFFIX): stream.cpp
	$(CXX) $(CXXFLAGS) stream.cpp -o $(TARGET)$(OUTPUT_SUFFIX) $(LDFLAGS)

clean:
	rm -f *.out *.m5


