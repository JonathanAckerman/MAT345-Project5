make:
	g++ -std=c++20 main.cpp -o proj5.exe
	./proj5.exe 784
clean:
	rm proj5.exe
	rm *.stackdump