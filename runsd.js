#!/usr/bin/env node

const fs = require('fs');
const util = require('util');
const exec = util.promisify(require('child_process').exec);
const readFile = util.promisify(fs.readFile);

async function run() {
    const models = ['runwaymlsd15/Resources', 'openjourneyv2/Resources', 'compiled'];
    let prompts = (await readFile('prompts.txt', 'utf8')).split('\n');
    let index = 0;

    while (index < prompts.length) {
        for (const model of models) {
            const prompt = prompts[index];
            console.log(`Running ${model} on prompt: ${prompt}...`);
            // run the specific command for the model on the prompt
            const { stdout, stderr } = await exec(`./StableDiffusionSample --resource-path ${model} "${prompt}" --step-count 120`);
            console.log(stdout);
            console.error(stderr);
            console.log('Done.');
            console.log();

            // check if any new prompts have been added to the file
            const updatedPrompts = (await readFile('prompts.txt', 'utf8')).split('\n');
            if (updatedPrompts.length > prompts.length) {
                prompts = updatedPrompts;
            }
        }
        index++;
    }
}

run().catch(error => console.error(error));
