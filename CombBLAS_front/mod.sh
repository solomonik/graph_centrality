sed -i -e 's/, MPIOp<_BinaryOperation, NT>::op(),/, __p,/g' ../*.cpp
sed -i -e 's/, MPIOp<_BinaryOperation, OUT>::op(),/, __p,/g' *.cpp
sed -i -e 's/.*MPI_.*, MPIOp<_BinaryOperation, OUT>::op(),.*/  MPI_Op __p = MPIOp<_BinaryOperation, OUT>::op();\n&/g' ../*.cpp
sed -i -e 's/.*MPI_.*, MPIOp<_BinaryOperation, NT>::op(),.*/  MPI_Op __p = MPIOp<_BinaryOperation, NT>::op();\n&/g' ../*.cpp
