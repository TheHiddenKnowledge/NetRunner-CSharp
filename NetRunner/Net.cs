using MathNet.Numerics.Data.Text;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;

/*
    NetRunner for C# 
    Created by Isaiah Finney
    This is a general purpose library that can create, execute, and optimize neural nets 
    The algorithm for gradient descent uses momentum to speed up the optimization process 
*/

namespace NetRunner
{
    public class Net
    {
        public int maxInput;
        public int maxOutput;
        public int[] layers;
        public List<int> totalNeurons = new List<int>();
        public Vector<double> inputs;
        public Vector<double> outputs;
        public List<Matrix<double>> weights = new List<Matrix<double>>();
        public List<Vector<double>> biases = new List<Vector<double>>();
        public List<Matrix<double>> gradientw = new List<Matrix<double>>();
        public List<Vector<double>> gradientb = new List<Vector<double>>();
        public List<Matrix<double>> prevgradientw;
        public List<Vector<double>> prevgradientb;
        // The output of each layer before the sigmoid function is applied 
        public List<Vector<double>> layerout = new List<Vector<double>>();
        // The output of each layer after the sigmoid function is applied 
        public List<Vector<double>> layersquish = new List<Vector<double>>();
        // Parameter for how much the sigmoid function squishes layer output 
        public double squish;
        // Parameters used for gradient descent speed 
        public double alpha;
        public double beta;
        public int successes = 0;
        public int fails = 0;
        Random random = new Random();
        // Constructor for the neural net class, where the dimensions of the inputs, outputs, and hidden layers are defined 
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
        }
        // Method that generates the weights and biases for the net, must be called after creating the net object 
        public void generatenet(double minweight, double maxweight, double minbias, double maxbias)
        {
            for (int i = 0; i < totalNeurons.Count - 1; i++)
            {
                // Creating the weight and biases matrices 
                Matrix<double> tempw = Matrix<double>.Build.Dense(totalNeurons[i + 1], totalNeurons[i]);
                Vector<double> tempb = Vector<double>.Build.Dense(totalNeurons[i + 1]);
                for (int j = 0; j < tempw.RowCount; j++)
                {
                    tempb[j] = (maxweight - minweight) * random.NextDouble() + minweight;
                    for (int k = 0; k < tempw.ColumnCount; k++)
                    {
                        tempw[j, k] = (maxbias - minbias) * random.NextDouble() + minbias;
                    }
                }
                weights.Add(tempw);
                biases.Add(tempb);
                // Creating the gradient matrices for the weights and biases 
                gradientw.Add(Matrix<double>.Build.Dense(totalNeurons[i + 1], totalNeurons[i]));
                gradientb.Add(Vector<double>.Build.Dense(totalNeurons[i + 1]));
            }
            prevgradientw = gradientw;
            prevgradientb = gradientb;
        }
        // Method for applying the sigmoid function to the input 
        public double sigmoid(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x / squish));
        }
        // Method for getting the derivative of the sigmoid function evaluated at the input 
        public double sigderiv(double x)
        {
            double num = Math.Pow(Math.E, x / squish) / Math.Pow((Math.Pow(Math.E, x / squish) + 1), 2) / squish;
            if (num == double.NaN)
            {
                double dz = .00001;
                return (sigmoid(x + dz) - sigmoid(x) / dz);
            }
            else
            {
                return num;
            }
        }
        // Method for evaluating the neural net with the provided input 
        public Vector<double> In2Out()
        {
            layerout = new List<Vector<double>>();
            layersquish = new List<Vector<double>>();
            Vector<double> current = inputs;
            Vector<double> currentsquish = inputs;
            layerout.Add(current);
            layersquish.Add(inputs);
            // Using matrix multiplication and addition to get each layers output 
            for (int i = 0; i < weights.Count; i++)
            {
                current = (weights[i] * current) + biases[i];
                currentsquish = Vector<double>.Build.DenseOfVector(current);
                layerout.Add(current);
                for (int j = 0; j < current.Count; j++)
                {
                    currentsquish[j] = sigmoid(currentsquish[j]);
                }
                layersquish.Add(currentsquish);
            }
            return currentsquish;
        }
        // Method that adjusts the output to match the desired range 
        public Vector<double> adjustoutput(double maxOut, double minOut)
        {
            return (maxOut - minOut) * outputs + minOut;
        }
        // Method that saves matrix data to a folder 
        public void savedata(string path)
        {
            Directory.CreateDirectory(path);
            string filename;
            string pathname;
            for (int i = 0; i < weights.Count; i++)
            {
                filename = "w" + i.ToString() + ".txt";
                pathname = Path.Combine(path, filename);
                DelimitedWriter.Write(pathname, weights[i]);
                filename = "b" + i.ToString() + ".txt";
                pathname = Path.Combine(path, filename);
                Matrix<double> tempb = Matrix<double>.Build.Dense(biases[i].Count, 1);
                for (int j = 0; j < biases[i].Count; j++)
                {
                    tempb[j, 0] = biases[i][j];
                }
                DelimitedWriter.Write(pathname, tempb);
            }
        }
        // Method that saves matrix data to a folder 
        public void loaddata(string path)
        {
            string filename;
            string pathname;
            for (int i = 0; i < weights.Count; i++)
            {
                filename = "w" + i.ToString() + ".txt";
                pathname = Path.Combine(path, filename);
                weights[i] = DelimitedReader.Read<double>(pathname, false);
                filename = "b" + i.ToString() + ".txt";
                pathname = Path.Combine(path, filename);
                Matrix<double> tempb = DelimitedReader.Read<double>(pathname, false);
                for (int j = 0; j < biases[i].Count; j++)
                {
                    biases[i][j] = tempb[j, 0];
                }
            }
        }
        // Method that loads matrix data from a folder 

        // Method that obtains the gradient with respect to the weights and biases 
        public void getgradient(Vector<double> expected)
        {
            List<List<double>> tempg = new List<List<double>>();
            for (int i = 0; i < weights.Count; i++)
            {
                int idx = weights.Count - i - 1;
                List<double> tempgg = new List<double>();
                for (int j = 0; j < weights[idx].RowCount; j++)
                {
                    // Using multivariable chain rule to get the gradient of the biases 
                    if (i == 0)
                    {
                        gradientb[idx][j] = sigderiv(layerout[idx + 1][j]) * 2 * (layersquish[idx + 1][j] - expected[j]);
                    }
                    else
                    {
                        gradientb[idx][j] = sigderiv(layerout[idx + 1][j]) * tempg[i - 1][j];
                    }
                    for (int k = 0; k < weights[idx].ColumnCount; k++)
                    {
                        // Using multivariable chain rule to get the gradient of the weights  
                        if (i == 0)
                        {
                            gradientw[idx][j, k] = layerout[idx][k] * sigderiv(layerout[idx + 1][j]) * 2 * (layersquish[idx + 1][j] - expected[j]);
                            if (j == 0)
                            {
                                tempgg.Add(weights[idx][j, k] * sigderiv(layerout[idx + 1][j]) * 2 * (layersquish[idx + 1][j] - expected[j]));
                            }
                            else
                            {
                                tempgg[k] += weights[idx][j, k] * sigderiv(layerout[idx + 1][j]) * 2 * (layersquish[idx + 1][j] - expected[j]);
                            }
                        }
                        else
                        {
                            gradientw[idx][j, k] = layerout[idx][k] * sigderiv(layerout[idx + 1][j]) * tempg[i - 1][j];
                            if (j == 0)
                            {
                                tempgg.Add(weights[idx][j, k] * sigderiv(layerout[idx + 1][j]) * tempg[i - 1][j]);
                            }
                            else
                            {
                                tempgg[k] += weights[idx][j, k] * sigderiv(layerout[idx + 1][j]) * tempg[i - 1][j];
                            }
                        }
                    }
                }
                tempg.Add(tempgg);
            }
        }
        // Method that gets the average gradient of a set of data and applies it to the weights and biases of the net 
        public double getset(List<Vector<double>> inputset, List<Vector<double>> expectedset)
        {
            // Average gradients for the set of data 
            List<Matrix<double>> avg_gw = new List<Matrix<double>>();
            List<Vector<double>> avg_gb = new List<Vector<double>>();
            // Average error for the set of data 
            double avg_error = 0;
            for (int i = 0; i < inputset.Count; i++)
            {
                inputs = inputset[i];
                In2Out();
                getgradient(expectedset[i]);
                for (int j = 0; j < weights.Count; j++)
                {
                    if (i == 0)
                    {
                        avg_gw.Add(gradientw[j]);
                        avg_gb.Add(gradientb[j]);
                    }
                    else
                    {
                        avg_gw[j] += gradientw[j];
                        avg_gb[j] += gradientb[j];
                    }
                }
                // Calculating the average error 
                double avg_expected = 0;
                for (int j = 0; j < expectedset[i].Count; j++)
                {
                    avg_expected += Math.Pow(layersquish[weights.Count][j] - expectedset[i][j], 2);
                }
                avg_expected /= expectedset[i].Count;
                avg_error += avg_expected;
            }
            avg_error /= expectedset.Count;
            // Descending the gradient using the momentum theorem for neural nets 
            for (int i = 0; i < weights.Count; i++)
            {
                weights[i] -= (beta * prevgradientw[i] + alpha * avg_gw[i]);
                biases[i] -= (beta * prevgradientb[i] + alpha * avg_gb[i]);
            }
            prevgradientb = avg_gb;
            prevgradientw = avg_gw;
            return avg_error;
        }
    }
}
