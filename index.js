const csv = require('csvtojson');
const shuffle = require('shuffle-array');
// const normalizerLib = require('neural-data-normalizer/dist/src/normalizer');
var synaptic = require('synaptic');
var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
    Architect = synaptic.Architect;

const csvFilePath = 'creditcard.csv';
let csvData = [];

console.log("Read data...");

csv()
    .fromFile(csvFilePath)
    .on('json', (jsonObj) => {
        csvData.push(jsonObj);
    })
    .on('done', () => {
        console.log("Dress data...");
        const data = dressData();
        
        console.log("Split data...");
        const [trainingData, testData] = splitData(data);

        console.log("Train network...");
        const network = train(trainingData);
        
        console.log("Test network...");
        test(network, testData);
    });

function dressData() {
    // csvData = csvData.slice(0,100);

    // const normalizer = new normalizerLib.Normalizer(csvData);
    // normalizer.setOutputProperties(['Class']);
    // normalizer.normalize();

    // const inputs = normalizer.getBinaryInputDataset();
    // const outputs = normalizer.getBinaryOutputDataset();

    // const trainingSet = [];
    // for(const i in inputs) {
    //     trainingSet.push({
    //         input: inputs[i],
    //         output: outputs[i]
    //     });
    // }

    const data = [];

    for (const entry of csvData) {
        data.push({
            input: [
                parseFloat(entry.V1), 
                parseFloat(entry.V2), 
                parseFloat(entry.V3), 
                parseFloat(entry.V4), 
                parseFloat(entry.V5), 
                parseFloat(entry.V6), 
                parseFloat(entry.V7), 
                parseFloat(entry.V8), 
                parseFloat(entry.V9), 
                parseFloat(entry.V10), 
                parseFloat(entry.V11), 
                parseFloat(entry.V12), 
                parseFloat(entry.V13), 
                parseFloat(entry.V14), 
                parseFloat(entry.V15), 
                parseFloat(entry.V16), 
                parseFloat(entry.V17), 
                parseFloat(entry.V18), 
                parseFloat(entry.V19), 
                parseFloat(entry.V20), 
                parseFloat(entry.V21), 
                parseFloat(entry.V22), 
                parseFloat(entry.V23), 
                parseFloat(entry.V24), 
                parseFloat(entry.V25), 
                parseFloat(entry.V26), 
                parseFloat(entry.V27), 
                parseFloat(entry.V28)
            ],
            output: [parseFloat(entry.Class)]
        })
    }

    return data;
}

function splitData(data) {
    shuffle(data);

    const numberOfTrainData = data.length * 0.8;

    const trainingData = data.slice(0, numberOfTrainData);
    const testData = data.slice(numberOfTrainData);

    return [trainingData, testData];
}

function train(data) {
    const network = new Architect.Perceptron(28, 50, 1);
    const trainer = new Trainer(network);

    var trainingOptions = {
        rate: .01,
        iterations: 20,
        error: .005,
        log: 1,
        shuffle: true,
        cost: Trainer.cost.CROSS_ENTROPY
      }
      
    trainer.train(data, trainingOptions);

    return network;
}

function test(network, data) {
    const frauds = data.filter(x => x.output[0] == 1);
    const valids = data.filter(x => x.output[0] == 0);

    console.log(network.activate(frauds[0].input));
    console.log(frauds[0].output);

    console.log(network.activate(valids[0].input));
    console.log(valids[0].output);

    const possibleFrauds = frauds.map(x => network.activate(x.input)[0]).filter(x => x > 0.7);
    console.log(`Frauds possible > 70%: ${possibleFrauds.length}/${frauds.length} (${possibleFrauds.length / frauds.length})`);

    const possibleValids = valids.map(x => network.activate(x.input)[0]).filter(x => x < 0.3);
    console.log(`Valids possible < 30%: ${possibleValids.length}/${valids.length} (${possibleValids.length / valids.length})`);
}