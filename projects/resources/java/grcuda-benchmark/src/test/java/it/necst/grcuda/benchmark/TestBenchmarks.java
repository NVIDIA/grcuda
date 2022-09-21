package it.necst.grcuda.benchmark;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.stream.JsonReader;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.io.*;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.text.Format;
import java.text.SimpleDateFormat;
import java.util.*;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeThat;
import static org.junit.Assume.assumeTrue;


public class TestBenchmarks{
    private String GRCUDA_HOME = System.getenv("GRCUDA_HOME");
    private String PATH;
    private GPU currentGPU;
    private String results_path;

    @Before
    public void init() throws IOException, InterruptedException {
        //create the folder to store the json results of the benchmarks
        PATH = GRCUDA_HOME+"/projects/resources/java/grcuda-benchmark/src/test/java/it/necst/grcuda/benchmark";
        Format formatter = new SimpleDateFormat("yyyy_MM_dd_hh_mm_ss");
        Date currentDate = new Date();
        String results_path = "./results/"+formatter.format(currentDate);
        this.results_path = results_path;

        int i=0;
        while(!new File(results_path).mkdirs()){
            results_path = "./results/"+formatter.format(currentDate)+"_("+i+")";
            i++;
        }


        // Compute BANDWIDTH MATRIX if necessary
        String BANDWIDTH_MATRIX_PATH = GRCUDA_HOME+"/projects/resources/connection_graph/datasets/connection_graph.csv";
        File f = new File(BANDWIDTH_MATRIX_PATH);
        if(!f.exists() && !f.isDirectory()) {
            // we need to compute the interconnection bandwidth matrix
            ProcessBuilder builder = new ProcessBuilder();
            builder.directory(new File(GRCUDA_HOME+"/projects/resources/connection_graph"));
            builder.command("bash -c ./run.sh".split("\\s+"));
            Process process = builder.start();
            int exitCode = process.waitFor();
            assertEquals("Return value should be 0", 0, exitCode);
        }

        // Get the model of GPUs installed in the system
        Set<GPU> detectedGPUS = new HashSet<>();
        ProcessBuilder builder = new ProcessBuilder();
        builder.command("nvidia-smi --query-gpu=gpu_name --format=csv".split("\\s+"));
        Process process = builder.start();
        int exitCode = process.waitFor();
        assertEquals("Return value should be 0", 0, exitCode);
        BufferedReader br=new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line;
        StringBuilder sb = new StringBuilder();
        br.readLine(); // discard "name" at the beginning of the output
        while((line=br.readLine())!=null){
            GPU g = GPU.valueOfName(line);
            assertNotEquals("GPU should be present in the GPU enum",null, g);
            detectedGPUS.add(g);
        }
        assertEquals(1, detectedGPUS.size());
        this.currentGPU = detectedGPUS.iterator().next();
    }

    @Test
    public void runAll_gtx1660_super() throws FileNotFoundException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException, JsonProcessingException {
        assumeTrue(this.currentGPU.equals(GPU.GTX1660_SUPER));

         // get the configuration for the selected GPU into a Config class
        String CONFIG_PATH = PATH + "/config_GTX1660_super.json";
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        JsonReader reader = new JsonReader(new FileReader(CONFIG_PATH));
        Config parsedConfig = gson.fromJson(reader, Config.class);
        //System.out.println(gson.toJson(parsedConfig)); // print the current configuration

        iterateAllPossibleConfig(parsedConfig);
    }

    @Test
    public void runAll_gtx960_multi() throws FileNotFoundException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException, JsonProcessingException {
        assumeTrue(this.currentGPU.equals(GPU.GTX960));

        // get the configuration for the selected GPU into a Config class
        String CONFIG_PATH = PATH + "/config_GTX960.json";
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        JsonReader reader = new JsonReader(new FileReader(CONFIG_PATH));
        Config parsedConfig = gson.fromJson(reader, Config.class);
        //System.out.println(gson.toJson(parsedConfig)); // print the current configuration

        iterateAllPossibleConfig(parsedConfig);
    }

    @Test
    public void runAll_V100_multi() throws FileNotFoundException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException, JsonProcessingException {
        assumeTrue(this.currentGPU.equals(GPU.V100));

        // get the configuration for the selected GPU into a Config class
        String CONFIG_PATH = PATH + "/config_V100.json";
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        JsonReader reader = new JsonReader(new FileReader(CONFIG_PATH));
        Config parsedConfig = gson.fromJson(reader, Config.class);
        System.out.println(gson.toJson(parsedConfig)); // print the current configuration

        iterateAllPossibleConfig(parsedConfig);
    }

    @Test
    public void runAll_A100_multi() throws FileNotFoundException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException, JsonProcessingException {
        assumeTrue(this.currentGPU.equals(GPU.A100));

        // get the configuration for the selected GPU into a Config class
        String CONFIG_PATH = PATH + "/config_A100.json";
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        JsonReader reader = new JsonReader(new FileReader(CONFIG_PATH));
        Config parsedConfig = gson.fromJson(reader, Config.class);
        //System.out.println(gson.toJson(parsedConfig)); // print the current configuration

        iterateAllPossibleConfig(parsedConfig);
    }

    /*
    This method reflects the pattern of benchmark_wrapper.py present in the python suite.
 */
    private void iterateAllPossibleConfig(Config parsedConfig) throws ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException, JsonProcessingException {
        String BANDWIDTH_MATRIX;
        ArrayList<String> dp, nsp, psp, cdp;
        ArrayList<Integer> ng, block_sizes;
        Integer nb; // number of blocks
        Integer blockSize1D, blockSize2D;
        int num_iter = parsedConfig.num_iter;

        Benchmark benchToRun;
        for(String bench : parsedConfig.benchmarks){ // given bench X from the set of all the benchmarks iterate over the number of elements associated with that benchmark
            ArrayList<Integer> sizes = parsedConfig.num_elem.get(bench);
            if(sizes == null) continue; //skip everything if no sizes are specified for the current bench
            for(Integer curr_size : sizes){ // given a specific input size iterate over the various execution policies
                for(String policy : parsedConfig.exec_policies){
                    if(policy.equals("sync")){
                        dp = new ArrayList<>(List.of(parsedConfig.dependency_policies.get(0)));
                        nsp = new ArrayList<>(List.of(parsedConfig.new_stream_policies.get(0)));
                        psp = new ArrayList<>(List.of(parsedConfig.parent_stream_policies.get(0)));
                        cdp = new ArrayList<>(List.of(parsedConfig.choose_device_policies.get(0)));
                        ng = new ArrayList<>(List.of(1));
                    }
                    else{
                        dp = parsedConfig.dependency_policies;
                        nsp = parsedConfig.new_stream_policies;
                        psp = parsedConfig.parent_stream_policies;
                        cdp = parsedConfig.choose_device_policies;
                        ng = parsedConfig.num_gpus;
                    }
                    for(int num_gpu : ng){
                        if(policy.equals("async") && num_gpu == 1){
                            dp = new ArrayList<>(List.of(parsedConfig.dependency_policies.get(0)));
                            nsp = new ArrayList<>(List.of(parsedConfig.new_stream_policies.get(0)));
                            psp = new ArrayList<>(List.of(parsedConfig.parent_stream_policies.get(0)));
                            cdp = new ArrayList<>(List.of(parsedConfig.choose_device_policies.get(0)));
                        }
                        else{
                            dp = parsedConfig.dependency_policies;
                            nsp = parsedConfig.new_stream_policies;
                            psp = parsedConfig.parent_stream_policies;
                            cdp = parsedConfig.choose_device_policies;
                        }
                        for(String m : parsedConfig.memory_advise){
                            for(Boolean p : parsedConfig.prefetch ){
                                for(Boolean s : parsedConfig.stream_attach){
                                    for(Boolean t : parsedConfig.time_computation){
                                        BANDWIDTH_MATRIX= GRCUDA_HOME+"/projects/resources/connection_graph/datasets/connection_graph.csv";
                                        for(String dependency_policy : dp){
                                            for(String new_stream_policy : nsp){
                                                for(String parent_stream_policy : psp){
                                                    for(String choose_device_policy : cdp){
                                                        BenchmarkConfig config = new BenchmarkConfig();

                                                        nb = parsedConfig.numBlocks.get(bench);
                                                        if(nb != null) config.numBlocks = nb;

                                                        blockSize1D = parsedConfig.block_size1d.get(bench);
                                                        if(blockSize1D != null) config.blockSize1D = blockSize1D;

                                                        blockSize2D = parsedConfig.block_size2d.get(bench);
                                                        if(blockSize2D != null) config.blockSize2D = blockSize2D;

                                                        config.debug = parsedConfig.debug;
                                                        config.benchmarkName = bench;
                                                        config.size = curr_size;
                                                        config.numGpus = num_gpu;
                                                        config.executionPolicy = policy;
                                                        config.dependencyPolicy = dependency_policy;
                                                        config.retrieveNewStreamPolicy = new_stream_policy;
                                                        config.retrieveParentStreamPolicy = parent_stream_policy;
                                                        config.deviceSelectionPolicy = choose_device_policy;
                                                        config.inputPrefetch = p;
                                                        config.totIter = num_iter;
                                                        config.forceStreamAttach = s;
                                                        config.memAdvisePolicy = m;
                                                        config.bandwidthMatrix = BANDWIDTH_MATRIX;
                                                        config.enableComputationTimers =t;
                                                        config.nvprof_profile = parsedConfig.nvprof_profile;
                                                        config.gpuModel = this.currentGPU.name;
                                                        config.results_path = this.results_path;
                                                        config.reInit = parsedConfig.reInit;

                                                        System.out.println(config);
                                                        benchToRun = createBench(config);
                                                        benchToRun.run();
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private Benchmark createBench(BenchmarkConfig config) throws ClassNotFoundException, NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException {
        // Courtesy of https://stackoverflow.com/questions/7495785/java-how-to-instantiate-a-class-from-string

        Class currBenchClass = Class.forName("it.necst.grcuda.benchmark.bench."+config.benchmarkName);

        Class[] types = {BenchmarkConfig.class};
        Constructor constructor = currBenchClass.getConstructor(types);

        Object[] parameters = {config};

        return (Benchmark) constructor.newInstance(parameters);

    }

}

enum GPU {
    GTX1660_SUPER("GeForce GTX 1660 SUPER"),
    A100("NVIDIA A100-SXM4-40GB"),
    V100("Tesla V100-SXM2-16GB"),
    GTX960("GeForce GTX 960");

    public final String name;

    GPU(String name){
        this.name = name;
    }

    public static GPU valueOfName(String toGet){
        for(GPU g : values()){
            if(g.name.equals(toGet))
                return g;
        }
        return null;
    }
}

/**
 * Used to map/parse the json config files to a class
 */
class Config {
    int num_iter;
    int heap_size;

    boolean reInit = false;
    boolean randomInit;
    boolean cpuValidation;
    boolean debug;
    boolean nvprof_profile;

    ArrayList<String> benchmarks;
    ArrayList<String> exec_policies;
    ArrayList<String> dependency_policies;
    ArrayList<String> new_stream_policies;
    ArrayList<String> parent_stream_policies;
    ArrayList<String> choose_device_policies;
    ArrayList<String> memory_advise;

    ArrayList<Boolean> prefetch;
    ArrayList<Boolean> stream_attach;
    ArrayList<Boolean> time_computation;

    ArrayList<Integer> num_gpus;

    HashMap<String, ArrayList<Integer>> num_elem;
    HashMap<String, Integer> numBlocks;
    HashMap<String, Integer> block_size1d;
    HashMap<String, Integer> block_size2d;

    @Override
    public String toString() {
        return "Config{" +
                "num_iter=" + num_iter +
                ", heap_size=" + heap_size +
                ", reInit=" + reInit +
                ", randomInit=" + randomInit +
                ", cpuValidation=" + cpuValidation +
                ", benchmarks=" + benchmarks +
                ", exec_policies=" + exec_policies +
                ", dependency_policies=" + dependency_policies +
                ", new_stream_policies=" + new_stream_policies +
                ", parent_stream_policies=" + parent_stream_policies +
                ", choose_device_policies=" + choose_device_policies +
                ", memory_advise=" + memory_advise +
                ", prefetch=" + prefetch +
                ", stream_attach=" + stream_attach +
                ", time_computation=" + time_computation +
                ", num_gpus=" + num_gpus +
                ", num_elem=" + num_elem +
                ", numBlocks=" + numBlocks +
                ", block_size1d=" + block_size1d +
                ", block_size2d=" + block_size2d +
                '}';
    }
}


