using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;
namespace NetRunner
{
    public class Net
    {
        public int maxInput;
        public int maxOutput;
        public int[] layers;
        public List<int> totalNeurons= new List<int>();
        public Vector<double> inputs;
        public Vector<double> outputs;
        public List<Matrix<double>> weights = new List<Matrix<double>>();
        public List<Vector<double>> biases = new List<Vector<double>>();
        public List<Matrix<double>> gradientw = new List<Matrix<double>>();
        public List<Vector<double>> gradientb = new List<Vector<double>>();
        public List<Matrix<double>> prevgradientw;
        public List<Vector<double>> prevgradientb;
        public List<Vector<double>> layerout = new List<Vector<double>>();
        public List<Vector<double>> layersquish = new List<Vector<double>>();
        public double squish;
        public double alpha;
        public double beta;
        public int successes = 0;
        public int fails = 0;
        public Net(int inputs, int outputs, int[] layers, double alpha, double beta, double squish)
        {
            maxInput = inputs;
            maxOutput = outputs;
            this.layers = layers;
            this.alpha = alpha;
            this.beta = beta;
            this.squish = squish;
            this.inputs = Vector<double>.Build.Dense(maxInput);
            this.outputs = Vector<double>.Build.Dense(maxOutput);
            totalNeurons.Add(maxInput);
            for (int i = 0; i < this.layers.Length; i++)
            {
                totalNeurons.Add(this.layers[i]);
            }
            totalNeurons.Add(maxOutput);
            for (int i = 0; i < totalNeurons.Count - 1; i++)
            {
                Matrix<double> tempw = 10 * Matrix<double>.Build.Random(totalNeurons[i + 1], totalNeurons[i]) - 5;
                weights.Add(tempw);
                Vector<double> tempb = 10 * Vector<double>.Build.Random(totalNeurons[i + 1]) - 5;
                biases.Add(tempb);
                gradientw.Add(Matrix<double>.Build.Dense(totalNeurons[i + 1], totalNeurons[i]));
                gradientb.Add(Vector<double>.Build.Dense(totalNeurons[i + 1]));
            }
            prevgradientw = gradientw;
            prevgradientb = gradientb;
        }
        public void RandomGenes()
        {
            weights = new List<Matrix<double>>();
            biases = new List<Vector<double>>();
            for (int i = 0; i < totalNeurons.Count - 1; i++)
            {
                Matrix<double> tempw = 10 * Matrix<double>.Build.Random(totalNeurons[i + 1], totalNeurons[i]) - 5;
                weights.Add(tempw);
                Matrix<double> tempb = 10 * Matrix<double>.Build.Random(totalNeurons[i + 1], 1) - 5;
                weights.Add(tempb);
            }
        }
        public double sigmoid(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x / squish));
        }
        public double sigderiv(double x)
        {
            return Math.Pow(Math.E, x / squish) / Math.Pow((Math.Pow(Math.E, x / squish) + 1), 2) / squish; 
        }
        public Vector<double> In2Out()
        {
            layerout = new List<Vector<double>>();
            layersquish = new List<Vector<double>>();
            Vector<double> current = inputs;
            layerout.Add(current);
            layersquish.Add(inputs);
            for(int i = 0; i < weights.Count; i++)
            {
                current = weights[i]* current + biases[i];
                Vector<double> currentsquish = Vector<double>.Build.DenseOfVector(current);
                layerout.Add(current); 
                for (int j = 0; j < current.Count; j++)
                {
                    currentsquish[j] = sigmoid(currentsquish[j]);
                }
                layersquish.Add(currentsquish);
            }
            return current;
        }
    }
}
