UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	LFLAGS=--shared -fPIC -rdynamic
	CFLAGS=-fPIC -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes
	LIB=/home/linyi2/anaconda3/lib
	INCLUDE=/home/linyi2/anaconda3/include/python3.6m
#	LIB=/home/linyi/anaconda3/lib
#	INCLUDE=/home/linyi/anaconda3/include/python3.8
endif
ifeq ($(UNAME_S),Darwin)
	LFLAGS=-bundle -undefined dynamic_lookup
	CFLAGS=-Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes
	LIB=/Users/lly/anaconda3/lib
	INCLUDE=/Users/lly/anaconda3/include/Python3.6m
endif

all: transform_kern.so


transform_kern.so: transform_kern.o
	g++ $(LFLAGS) -L$(LIB) $^ -o $@

%.o: %.cpp %.h
	gcc $(CFLAGS) -I$(INCLUDE) -o $@ -c $<

clean:
	rm -rf *.o *.so