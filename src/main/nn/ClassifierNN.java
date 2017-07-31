package main.nn;

import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.text.Normalizer;
import java.util.ArrayList;

/**
 * Created by Ross Byrne on 27/07/17.
 */
public class ClassifierNN {

    private static Logger log = LoggerFactory.getLogger(ClassifierNN.class);

    public static void main(String[] args) throws  Exception {

        double[][] data =

            { //Health, Sword, Bomb, Enemies, class

                    // No Sword, No Bomb
                    { 1, 0, 0, 0, 0 }, { 1, 0, 0, 0.5, 0 }, { 1, 0, 0, 1, 2 }, // full health, enemies covered
                    { 0.5, 0, 0, 0, 1 }, { 0.5, 0, 0, 0.5, 2 }, { 0.5, 0, 0, 1, 3 }, // minior injuries, enemies covered
                    { 0, 0, 0, 0, 2 }, { 0, 0, 0, 0.5, 3 }, { 0, 0, 0, 1, 3 }, // serious injuries, enemies covered

                    // Sword, No Bomb
                    { 1, 1, 0, 0, 0 }, { 1, 1, 0, 0.5, 0 }, { 1, 1, 0, 1, 0 }, // full health, enemies covered
                    { 0.5, 1, 0, 0, 0 }, { 0.5, 1, 0, 0.5, 2 }, { 0.5, 1, 0, 1, 1 }, // minior injuries, enemies covered
                    { 0, 1, 0, 0, 2 }, { 0, 1, 0, 0.5, 3 }, { 0, 1, 0, 1, 3 }, // serious injuries, enemies covered

                    // No Sword, Bomb
                    { 1, 0, 1, 0, 0 }, { 1, 0, 1, 0.5, 0 }, { 1, 0, 1, 1, 0 }, // full health, enemies covered
                    { 0.5, 0, 1, 0, 0 }, { 0.5, 0, 1, 0.5, 1 }, { 0.5, 0, 1, 1, 1 }, // minior injuries, enemies covered
                    { 0, 0, 1, 0, 2 }, { 0, 0, 1, 0.5, 2 }, { 0, 0, 1, 1, 3 }, // serious injuries, enemies covered

                    // Sword, Bomb
                    { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0.5, 0 }, { 1, 1, 1, 1, 0 }, // full health, enemies covered
                    { 0.5, 1, 1, 0, 0 }, { 0.5, 1, 1, 0.5, 0 }, { 0.5, 1, 1, 1, 1 }, // minior injuries, enemies covered
                    { 0, 1, 1, 0, 0 }, { 0, 1, 1, 0.5, 2 }, { 0, 1, 1, 1, 2 } // serious injuries, enemies covered
            };

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new File("resources/combatTrainingData.csv")));

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 4;     //4 classes. Classes have integer values 0, 1, 2 or 3
        int batchSize = 50;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
      //  SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(.9);  //Use 65% of data for training

        DataSet trainingData = allData;
        //DataSet testData = testAndTrain.getTest();

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
       // normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set


        final int numInputs = 4;
        int outputNum = 4;
        int iterations = 1000;
        int epochs = 10;
        long seed = 6;

        // create an INDArray from data 2dim array
        //INDArray d = Nd4j.create(data);


        log.info("Build model....");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.1)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(500));

        // train for number of epochs
        for(int i = 0; i < epochs; i++) {

            // train neural network
            model.fit(trainingData);

        } // if


        //System.out.println(testData.getFeatureMatrix());

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(4);
        INDArray output = model.output(trainingData.getFeatureMatrix());
        eval.eval(trainingData.getLabels(), output);

        log.info(eval.stats());

        //System.out.println(trainingData.getFeatureMatrix());
        //System.out.println(output);

        System.out.println("\nTesting Inputs\n");

        INDArray inputs = Nd4j.create(new double[] {1, 0, 0, 0}); // create inputs
        System.out.println(getNetworkOutput(model, normalizer, inputs)); // expected result: 0

        inputs = Nd4j.create(new double[] {0, 1, 0, 0.5});
        System.out.println(getNetworkOutput(model, normalizer, inputs)); // expected result: 3

        inputs = Nd4j.create(new double[] {0.5, 0, 1, 1}); // expected result: 1
        System.out.println(getNetworkOutput(model, normalizer, inputs));

        inputs = Nd4j.create(new double[] {0, 1, 0, 0}); // expected result: 2
        System.out.println(getNetworkOutput(model, normalizer, inputs));


    } // main()

    // method for normalising input, running it through the neural net and returning the output
    private static INDArray getNetworkOutput(MultiLayerNetwork net, DataNormalization normalizer, INDArray input){

        // normalise inputs
        normalizer.transform(input);

        // run input through Neural Net
        return net.output(input);

    } // getModelOutput()


} // class
