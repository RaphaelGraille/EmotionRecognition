#ifndef TESTSVM_H
#define TESTSVM_H

void trainAndTest(int nbTrain);
void crossValidation(int K, std::string DescAndLabels, std::string resPath);

#endif