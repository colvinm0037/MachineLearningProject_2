package com.micah.ml;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.SingleCrossOver;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

public class FourPeaks {

    
    private static final int N = 500;    
    private static final int T = N / 5;
	
	public static void runFourPeaks() {
		
		
		System.out.println("Running Four Peaks Tests");
		
		int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        NumberFormat formatter = new DecimalFormat("#0.00");     
        
        runRHC(hcp, ef);
        runSA(hcp, ef);
        
        // runGA();
        // runMIMIC();
	        

		
	}
	
	
	private static void runRHC(HillClimbingProblem hcp, EvaluationFunction ef) {
		
		System.out.println("Beginning Randomized Hill Climbing");
		
		 // RHC over various iteration amounts
        int iterations = 10000;
        while (iterations <= 300000) {
        	RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
	        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterations);
	        fit.train();
	        iterations += 10000;
	        System.out.println(ef.value(rhc.getOptimal()));
        }

        //  What is the best it can do in under 30 seconds
        Queue<Double> q = new LinkedList<Double>();
        iterations = 10000;
        
        long timeTaken = 0;
        while (timeTaken < 30000) {
        	long currentTime = System.currentTimeMillis();
        	RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
	        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterations);
	        fit.train();
	        timeTaken = System.currentTimeMillis() - currentTime;

	        System.out.println(ef.value(rhc.getOptimal()) + ", Time Take = " + timeTaken + ", Iterations: " + iterations);
	        iterations += 10000;
	        double result = ef.value(rhc.getOptimal());
        	boolean convergence = true;
	        if (q.size() == 5) {
	        	q.remove();	        	
	        } 
	        q.add(result);
	        for (double d : q) {
	        	if (Math.abs(d - N) > 1) {
	        		convergence = false;
	        		break;
	        	}
	        }
	        
	        if (convergence) break;
	        
        }
		
	}
	
	private static void runSA(HillClimbingProblem hcp, EvaluationFunction ef) {
		
		System.out.println("Beginning SA");
		
		// Various Cooling values
        double value = 0.02;
        int iterations = 200000;
        for (double k = .02; k < 1; k +=.02) {
	        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, value, hcp);
	        FixedIterationTrainer fit = new FixedIterationTrainer(sa, iterations);
	        fit.train();
	        System.out.println(ef.value(sa.getOptimal()));
	        value += .02;	        
        }

        // Various iteration amounts
        iterations = 10000;
        while (iterations <= 300000) {
	        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
	        FixedIterationTrainer fit = new FixedIterationTrainer(sa, iterations);
	        fit.train();
	        System.out.println(ef.value(sa.getOptimal()));
	        iterations += 10000;
        }

        // How far can it get in 30 seconds
        Queue<Double> q = new LinkedList<Double>();
        long timeTaken = 0;
        while (timeTaken < 30000) {
        	long currentTime = System.currentTimeMillis();
        	SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
	        FixedIterationTrainer fit = new FixedIterationTrainer(sa, iterations);
	        fit.train();
	        timeTaken = System.currentTimeMillis() - currentTime;

	        System.out.println(ef.value(sa.getOptimal()) + ", Time Take = " + timeTaken + ", Iterations: " + iterations);
	        iterations += 10000;
	        double result = ef.value(sa.getOptimal());
        	boolean convergence = true;
	        if (q.size() == 5) {
	        	q.remove();	        	
	        } 
	        q.add(result);
	        for (double d : q) {
	        	if (Math.abs(d - N) > 1) {
	        		convergence = false;
	        		break;
	        	}
	        }
	        
	        if (convergence) break;	        
        }
	}
	
	private static void runGA(GeneticAlgorithmProblem gap, ProbabilisticOptimizationProblem pop, EvaluationFunction ef) {
        
		int iterations = 10000;
		
		// TODO: Need to get results for GA
        while (iterations <= 300000) {

	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(1000, 500, 250, gap);
	        FixedIterationTrainer fit = new FixedIterationTrainer(ga, iterations);
	        fit.train();
	        System.out.println(ef.value(ga.getOptimal()));
	        iterations += 10000;
        }
    
	}
	
	private static void runMIMIC() {
//		MIMIC mimic = new MIMIC(200, 20, pop);
//	    FixedIterationTrainer fit = new FixedIterationTrainer(mimic, iterations);
//	    fit.train();
//	    System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
	    
	}
	
}
