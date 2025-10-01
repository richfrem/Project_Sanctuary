// ast_utilities/js_transformer/transform.js
const fs = require('fs');
const esprima = require('esprima');
const escodegen = require('escodegen');
const estraverse = require('estraverse');

function applyTransformation(sourceCode, instruction) {
    const ast = esprima.parseScript(sourceCode);

    estraverse.traverse(ast, {
        enter: function (node, parent) {
            // This is where we will implement specific transformation logic.
            // For our first test, we'll replace a specific variable's value.

            if (instruction.type === 'REPLACE_VARIABLE_STRING_VALUE' &&
                node.type === 'VariableDeclarator' &&
                node.id.name === instruction.variableName) {

                // Target the 'value' of the variable's 'init' property.
                if (node.init && node.init.type === 'Literal') {
                    console.log(`[AST] Found variable '${instruction.variableName}'. Replacing value.`);
                    node.init.value = instruction.newValue;
                    node.init.raw = `'${instruction.newValue}'`; // Update the raw representation as well
                }
            }
        }
    });

    return escodegen.generate(ast);
}

function main() {
    const [sourcePath, instructionJson] = process.argv.slice(2);

    if (!sourcePath || !instructionJson) {
        console.error('Usage: node transform.js <path_to_source_file> \'<json_instruction>\'');
        process.exit(1);
    }

    try {
        const sourceCode = fs.readFileSync(sourcePath, 'utf-8');
        const instruction = JSON.parse(instructionJson);

        const transformedCode = applyTransformation(sourceCode, instruction);

        // Output the transformed code to stdout. The calling Python script will capture this.
        process.stdout.write(transformedCode);

    } catch (e) {
        console.error(`[AST-TRANSFORMER-ERROR] ${e.message}`);
        process.exit(1);
    }
}

main();