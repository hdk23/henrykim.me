# CS50 Lab 3
## CS50 Summer 2021

### Functionality test

Various functionality tests are conducted in bagtest.c: 

* TEST 1: creating an empty bag
* TEST 2: inserting nodes into a valid bag
* TEST 3: inserting nodes into a null bag
* TEST 4: inserting null item into a valid bag
* TEST 5: inserting null item into a null bag
* TEST 6: printing nodes in a valid bag
* TEST 7: deleting a valid bag


#### Functionality test output
```bash
$ make clean; make test
./bagtest < test.names
Count (should be zero): 0
testing bag_insert...
Count (should be 66): 66
test with null bag, good item...
test with null item...
test with null bag, null item...
The bag:
{"Austin","Ziyi","Katrina","Carlos","Xiaoyi","Joseph","Noah","David","Naina","Yunjin","Alexander","James","Samuel","Jai","Andrew","Nathan","Rory","William","Alex","Emma","Naren","John","Joseph","Christian","William","Jiro","Dhaivat","John","Isaiah","Bryan","Rohith","Chirag","Jiehui","Joanna","Amber","Sydney","Seungjae","Grant","Dante","Yash","Janvi","Aadil","Ryan","Katelyn","Ian","Callum","Robert","Vivek","Shijie","Paul","Sanjana","Cara","Daniel","Quang","Jared","Sylvester","Aditya","Urie","Raymond","Jiuqi","Sunbir","Jose","Zachary","makale","Anne","Yunive",}
delete the bag...
```

The above output excludes the compilation output with `gcc`.


### Memory test
```bash
$ make clean; make valgrind
==37729== 
==37729== HEAP SUMMARY:
==37729==     in use at exit: 0 bytes in 0 blocks
==37729==   total heap usage: 136 allocs, 136 frees, 15,707 bytes allocated
==37729== 
==37729== All heap blocks were freed -- no leaks are possible
==37729== 
==37729== For lists of detected and suppressed errors, rerun with: -s
==37729== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```

The above output excludes the compilation output and the test program's normal output.



