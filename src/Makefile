LOG_PATH   =../../log/output/
BOOST_PATH =../../../local/boost/

INCLUDEDIR= \
		 -I. \
		 -I$(BOOST_PATH)/include \
		 -I$(LOG_PATH)/include \

LIBDIR=	\
		-L$(LOG_PATH)/lib -llog \
		-lpthread -lm -lstdc++

GCC = g++

SRCWILD    = $(wildcard *.cpp)
OBJS       = $(patsubst %.cpp, %_at.o, $(SRCWILD))

all : clean compile

ifeq ($(OPT), release)
CPPFLAGS       = -g -O3 -finline-functions -Wall -Winline -pipe -Wno-deprecated -Wunused-variable -DNDEBUG
endif

ifeq ($(OPT), debug)
CPPFLAGS = -g -finline-functions -Wall -Winline -pipe -Wno-deprecated -DMAXENT_DEBUG
endif

compile : $(OBJS)


clean:
	rm -rf $(OBJS)

%_at.o	: %.cpp
	$(GCC) $(CPPFLAGS) -c $< -o $@ $(INCLUDEDIR)

