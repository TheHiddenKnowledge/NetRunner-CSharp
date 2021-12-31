using MathNet.Numerics.LinearAlgebra;
using NetRunner;
using System;
using System.Collections.Generic;
namespace NetRunnerDebug
{
    class Program
    {
        // This example will take in an array of booleans and then determine the color using a neural net 
        static void Main(string[] args)
        {
            // Creating neural net object 
            Net net = new Net(3, 8, new int[] { 6, 8 }, .0001, .9, 2.5);
            // Generating a random net to start off from 
            net.generatenet(-.5, .5, -.5, .5);
            double error = 1;
            int i = 0;
            Random random = new Random();
            // The net will continue to descend until the error is below the threshold value 
            while (error > .00001)
            {
                i += 1;
                List<Vector<double>> expectedset = new List<Vector<double>>();
                List<Vector<double>> inputset = new List<Vector<double>>();
                for (int j = 0; j < 10; j++)
                {
                    // Generating RGB boolean inputs
                    double[] color = new double[] { random.Next(2), random.Next(2), random.Next(2) };
                    inputset.Add(Vector<double>.Build.DenseOfArray(color));
                    // Generating expected outputs from the given inputs 
                    if (color[0] == 1 && color[1] == 0 && color[2] == 0)
                    {
                        double[] expected = new double[] { 1, 0, 0, 0, 0, 0, 0, 0 };
                        expectedset.Add(Vector<double>.Build.DenseOfArray(expected));
                    }
                    else if (color[0] == 0 && color[1] == 1 && color[2] == 0)
                    {
                        double[] expected = new double[] { 0, 1, 0, 0, 0, 0, 0, 0 };
                        expectedset.Add(Vector<double>.Build.DenseOfArray(expected));
                    }
                    else if (color[0] == 0 && color[1] == 0 && color[2] == 1)
                    {
                        double[] expected = new double[] { 0, 0, 1, 0, 0, 0, 0, 0 };
                        expectedset.Add(Vector<double>.Build.DenseOfArray(expected));
                    }
                    else if (color[0] == 1 && color[1] == 1 && color[2] == 0)
                    {
                        double[] expected = new double[] { 0, 0, 0, 1, 0, 0, 0, 0 };
                        expectedset.Add(Vector<double>.Build.DenseOfArray(expected));
                    }
                    else if (color[0] == 1 && color[1] == 0 && color[2] == 1)
                    {
                        double[] expected = new double[] { 0, 0, 0, 0, 1, 0, 0, 0 };
                        expectedset.Add(Vector<double>.Build.DenseOfArray(expected));
                    }
                    else if (color[0] == 0 && color[1] == 1 && color[2] == 1)
                    {
                        double[] expected = new double[] { 0, 0, 0, 0, 0, 1, 0, 0 };
                        expectedset.Add(Vector<double>.Build.DenseOfArray(expected));
                    }
                    else if (color[0] == 1 && color[1] == 1 && color[2] == 1)
                    {
                        double[] expected = new double[] { 0, 0, 0, 0, 0, 0, 1, 0 };
                        expectedset.Add(Vector<double>.Build.DenseOfArray(expected));
                    }
                    else if (color[0] == 0 && color[1] == 0 && color[2] == 0)
                    {
                        double[] expected = new double[] { 0, 0, 0, 0, 0, 0, 0, 1 };
                        expectedset.Add(Vector<double>.Build.DenseOfArray(expected));
                    }
                }
                error = net.getset(inputset, expectedset);
                if (i % 500 == 0)
                {
                    Console.WriteLine("The error after " + i + " iterations is " + error);
                }
            }
            Console.WriteLine("The error after " + i + " iterations is " + error);
            Console.WriteLine();
            Vector<double> input;
            // Testing to see how accurate the neural net is at guessing the colors 
            List<double[]> colors = new List<double[]>() { new double[] { 1, 0, 0 }, new double[] { 0, 1, 0 }, new double[] { 0, 0, 1 }, new double[] { 1, 1, 0 },
                new double[] { 1, 0, 1 }, new double[] { 0, 1, 1 }, new double[] { 1, 1, 1 }, new double[] { 0, 0, 0 }, };
            string[] colornames = { "Red", "Green", "Blue", "Yellow", "Purple", "Cyan", "White", "Black" };
            for (int k = 0; k < colors.Count; k++)
            {
                input = Vector<double>.Build.DenseOfArray(colors[k]);
                net.inputs = input;
                net.outputs = net.In2Out();
                Console.WriteLine(colornames[net.outputs.MaximumIndex()]);
            }
            net.savedata(@"c:\BestNet");
            Console.ReadLine();
        }
    }
}
