using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
namespace NetRunner
{
    public class Net
    {
        public double fitness = 0;
        public double maxfitness = 0;
        double neuronrate = .5f;
        double synapserate = .55f;
        double weightrate = .55f;
        int generation = 0;
        int maxInput;
        int maxOutput;
        int maxNeuron;
        int layers;
        public double[] inputs;
        public double[] outputs;
        public List<List<List<double>>> weights = new List<List<List<double>>>();
        public List<List<List<double>>> maxweights = new List<List<List<double>>>();
        public List<List<List<int>>> synapses = new List<List<List<int>>>();
        public List<List<List<int>>> maxsynapses = new List<List<List<int>>>();
        public List<List<int>> paths = new List<List<int>>();
        public List<List<int>> maxpaths = new List<List<int>>();
        public List<int> totalNeurons = new List<int>();
        public int successes = 0;
        public int fails = 0;
        Random rand = new Random();
        public Net(int inputs, int outputs, int layers, int maxNeuron)
        {
            this.maxInput = inputs;
            this.maxOutput = outputs;
            this.maxNeuron = maxNeuron;
            this.layers = layers;
            this.inputs = new double[maxInput];
            this.outputs = new double[maxOutput];
            this.totalNeurons.Add(this.maxInput);
            for(int i = 0; i < layers; i++) {
                this.totalNeurons.Add(this.maxNeuron);
            }
            this.totalNeurons.Add(this.maxOutput);
        }
        public void Path2Syn()
        {
            for(int i = 0; i<this.paths.Count; i++)
            {
                for(int j = 0; j < this.paths[i].Count-1; j++)
                {
                    int idx1 = this.paths[i][j];
                    int idx2 = this.paths[i][j + 1];
                    this.synapses[j][idx1][idx2] = 1;
                }
            }
        }
        public void RandomGenes()
        {
            this.weights = new List<List<List<double>>>();
            this.synapses = new List<List<List<int>>>();
            this.paths = new List<List<int>>();
            for(int i = 0; i < this.totalNeurons.Count - 1; i++)
            {
                List<List<int>> temps = new List<List<int>>();
                List<List<double>> tempw = new List<List<double>>();
                for (int j = 0; j < this.totalNeurons[i]; j++)
                {
                    List<int> tempn = new List<int>();
                    List<double> tempww = new List<double>();
                    for (int k = 0; k < this.totalNeurons[i+1]; k++)
                    {
                        tempn.Add(0);
                        tempww.Add(2 * (rand.NextDouble()-.5f));
                    }
                    temps.Add(tempn);
                    tempw.Add(tempww);
                }
                this.synapses.Add(temps);
                this.weights.Add(tempw);
            }
            for(int i = 0; i < this.totalNeurons[0]; i++)
            {
                List<int> tempp = new List<int>();
                int num = rand.Next(1, this.totalNeurons[1]+1); 
;                for (int j = 0; j < num; j++)
                {
                    int rand1 = rand.Next(0, this.totalNeurons[1]);
                    tempp.Add(i);
                    tempp.Add(rand1);
                    for (int k = 0; k < this.totalNeurons.Count - 2; k++)
                    {
                        int rand2 = rand.Next(0, this.totalNeurons[k + 2]);
                        tempp.Add(rand2);
                        if (!this.paths.Contains(tempp))
                        {
                            this.paths.Add(tempp);
                        }
                    }
                    tempp = new List<int>();
                }
            }
            Path2Syn();
        }
        public void In2Out()
        {
            double[] layerout = this.inputs;
            for(int i = 0; i < this.synapses.Count; i++)
            {
                double[] templ = new double[this.totalNeurons[i + 1]];
                for (int j = 0; j < this.synapses[i].Count; j++)
                {
                    for (int k = 0; k < this.synapses[i][j].Count; k++)
                    {
                        if (this.synapses[i][j][k] == 1)
                        {
                            if (templ[k] == 0)
                            {
                                templ[k] = layerout[j] * this.weights[i][j][k];
                            }
                            else
                            {
                                templ[k] += layerout[j] * this.weights[i][j][k];
                            }
                        }
                    }
                }
                layerout = templ;
            }
            this.outputs = layerout;
        }
        public void MutateSynapse()
        {
            this.synapses = new List<List<List<int>>>();
            for (int i = 0; i < this.totalNeurons.Count - 1; i++)
            {
                List<List<int>> temps = new List<List<int>>();
                for (int j = 0; j < this.totalNeurons[i]; j++)
                {
                    List<int> tempn = new List<int>();
                    for (int k = 0; k < this.totalNeurons[i + 1]; k++)
                    {
                        tempn.Add(0);
                    }
                    temps.Add(tempn);
                }
                this.synapses.Add(temps);
            }
            if (rand.NextDouble() < .75)
            {
                int idx1 = rand.Next(0, this.paths.Count);
                int idx2 = rand.Next(0, this.paths[idx1].Count);
                int rand1 = rand.Next(0, this.totalNeurons[idx2]);
                List<int> current = this.paths[idx1];
                current[idx2] = rand1;
                if (!this.paths.Contains(current))
                {
                    this.paths[idx1] = current;
                }
            }
            if(rand.NextDouble() < .25)
            {
                List<int> tempp = new List<int>();
                for(int i = 0; i < this.totalNeurons.Count; i++)
                {
                    int rand1 = rand.Next(0, this.totalNeurons[i]);
                    tempp.Add(rand1);
                }
                if (!this.paths.Contains(tempp))
                {
                    this.paths.Add(tempp);
                }
            }
            if(rand.NextDouble() < .25)
            {
                int rand1 = rand.Next(0, this.paths.Count);
                if(this.paths.Count > this.maxOutput * this.maxInput * this.maxNeuron * this.layers / 4)
                {
                    this.paths.RemoveAt(rand1);
                }
            }
            Path2Syn();
        }
        public void MutateWeight()
        {
            int i = rand.Next(0, this.synapses.Count);
            int num = rand.Next(1, this.synapses[i].Count + 1);
            for (int j = 0; j < num; j++)
            {
                int idx1 = rand.Next(0, this.synapses[i].Count);
                int idx2 = rand.Next(0, this.synapses[i][idx1].Count);
                this.weights[i][idx1][idx2] = 2 * (rand.NextDouble() - .5);
            }
        }
        public void Mutate()
        {
            double p1 = rand.NextDouble();
            double p2 = rand.NextDouble();
            if (p1 < this.synapserate)
            {
                MutateSynapse();
            }
            if (p2 < this.weightrate)
            {
                MutateWeight();
            }
        }
        public void CheckFitness()
        {
            if (this.fitness > this.maxfitness)
            {
                this.maxfitness = this.fitness;
                this.maxsynapses = this.synapses;
                this.maxweights = this.weights;
                this.maxpaths = this.paths;
            }
            else
            {
                this.synapses = this.maxsynapses;
                this.weights = this.maxweights;
                this.paths = this.maxpaths;
            }
        }
    }
}
