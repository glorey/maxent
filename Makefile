LOG_PATH   =../log/output/
BOOST_PATH =../../local/boost/

INCLUDEDIR=  \
		 -I. \
		 -I$(BOOST_PATH)/include \
		 -I$(LOG_PATH)/include \

LIBDIR=	\
		-L$(LOG_PATH)/lib -llog \
		-lpthread -lm -lstdc++

GCC = g++
AR  = ar

#OPT = release
OPT = debug

ifeq ($(OPT), release)
	CPPFLAGS = -g -finline-functions -Wall -Winline -pipe -Wno-deprecated -Wunused-variable -fPIC
endif

ifeq ($(OPT), debug)
	CPPFLAGS = -g -finline-functions -Wall -Winline -pipe -Wno-deprecated -fPIC -DMAXENT_DEBUG
endif

CPPFLAGS_DEBUG = -g -finline-functions -Wall -Winline -pipe -Wno-deprecated -fPIC

LIBS    = libmaxent.a
TARGET1 = predict
TARGET2 = train

INCLUDES = src/*.h

OBJS = src/*_at.o

OBJ1    = bin/predict_at.o
OBJ2    = bin/train_at.o

all : clean output

compile:
	make compile -j4 -C src OPT=$(OPT)
	make compile -j4 -C bin OPT=$(OPT)

$(OBJS) $(OBJ1) $(OBJ2) : compile

output: $(LIBS) $(TARGET1) $(TARGET2)
	mkdir -p ./output/include
	mkdir -p ./output/bin
	mkdir -p ./output/lib
	cp -f ${LIBS} ./output/lib
	cp -f $(INCLUDES) ./output/include
	cp -f $(TARGET1) ./output/bin
	cp -f $(TARGET2) ./output/bin
	cp -r demo ./output
	rm -fr $(LIBS)
	rm -fr $(TARGET1)
	rm -fr $(TARGET2)

$(TARGET1) : $(OBJ1) $(OBJS)
	$(GCC) -g -o $@ $^ $(LIBDIR)

$(TARGET2) : $(OBJ2) $(OBJS)
	$(GCC) -g -o $@ $^ $(LIBDIR)

$(LIBS) : $(OBJS)
	$(AR) rcv $@ $^

clean:
	rm -f $(LIBS) $(TARGET1) $(TARGET2)
	make clean -C src
	make clean -C bin
	rm -rf ./output
	rm -rf sample/log sample/$(TARGET1)

