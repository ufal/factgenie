function getSelectedDatasets() {
    const selectedDatasets = $('.btn-check-dataset:checked').map(function () {
        return $(this).data('content');
    }).get();
    return selectedDatasets;
}

function getSelectedSplits() {
    const selectedSplits = $('.btn-check-split:checked').map(function () {
        return $(this).data('content');
    }).get();
    return selectedSplits;
}

function addDatasetButton(dataset) {
    $('#datasets-container').append(`
    <div class="form-check form-switch">
        <input class="form-check-input btn-check-dataset" type="checkbox" role="switch"
            id="btn-check-dataset-${dataset}" data-content="${dataset}">
        <label class="form-check-label btn-check-dataset-label" for="btn-check-dataset-${dataset}">${dataset}</label>
    </div>
`);
}


function addSplitButton(split) {
    if ($('#splits-container').find(`#btn-check-split-${split}`).length > 0) {
        return;
    }
    $('#splits-container').append(`
    <div class="form-check form-switch">
        <input class="form-check-input btn-check-split" type="checkbox" role="switch"
            id="btn-check-split-${split}" data-content="${split}">
        <label class="form-check-label btn-check-split-label" for="btn-check-split-${split}">${split}</label>
    </div>
`);
}

function addOutputButton(output) {
    if ($('#outputs-container').find(`#btn-check-output-${output}`).length > 0) {
        return;
    }
    $('#outputs-container').append(`
        <div class="form-check form-switch">
            <input class="form-check-input btn-check-output" type="checkbox" role="switch"
                id="btn-check-output-${output}" data-content="${output}">
            <label class="form-check-label btn-check-output-label" for="btn-check-output-${output}">${output}</label>
        </div>
    `);
}


function gatherComparisonData() {
    var campaign_datasets = [];
    var campaign_splits = [];
    var campaign_outputs = [];

    $(".btn-check-dataset").each(function () {
        if ($(this).prop("checked")) {
            campaign_datasets.push($(this).attr("data-content"));
        }
    });
    $(".btn-check-split").each(function () {
        if ($(this).prop("checked")) {
            campaign_splits.push($(this).attr("data-content"));
        }
    });
    $(".btn-check-output").each(function () {
        if ($(this).prop("checked")) {
            campaign_outputs.push($(this).attr("data-content"));
        }
    });
    var combinations = [];

    if (mode == "llm_eval" || mode == "crowdsourcing") {
        // Select all the available combinations according to the selection
        combinations = available_data.filter(function (model_out) {
            return campaign_datasets.includes(model_out.dataset) && campaign_splits.includes(model_out.split) && campaign_outputs.includes(model_out.setup_id);
        });
        // Remove duplicates
        combinations = combinations.filter((v, i, a) => a.findIndex(t => (t.dataset === v.dataset && t.split === v.split && t.setup_id === v.setup_id)) === i);

    } else if (mode == "llm_gen") {
        // Select all the available combinations according to the selection
        combinations = available_data.filter(function (model_out) {
            return campaign_datasets.includes(model_out.dataset) && campaign_splits.includes(model_out.split);
        });
        // Remove duplicates
        combinations = combinations.filter((v, i, a) => a.findIndex(t => (t.dataset === v.dataset && t.split === v.split)) === i);
    }

    return combinations;
}


function sortCheckboxes(container) {
    // Sort all the checkboxes in the given container alphabetically
    const checkboxes = container.find('.form-check-input');
    checkboxes.sort(function (a, b) {
        return $(a).data('content').localeCompare($(b).data('content'));
    });
    container.empty();
    checkboxes.each(function () {
        container.append($(this).parent());
    });
}

function populateDatasets() {
    for (const dataset of Object.keys(datasets)) {
        addDatasetButton(dataset);
    }
    sortCheckboxes($('#datasets-container'));
}



function updateComparisonData() {
    const selectedDatasets = getSelectedDatasets();
    const selectedSplits = getSelectedSplits();

    if (selectedDatasets.length === 0) {
        $('#splits-column').hide();
        $('#outputs-column').hide();
    } else {
        $('#splits-column').show();
    }
    if (selectedSplits.length === 0 || selectedDatasets.length === 0) {
        $('#outputs-column').hide();
    } else if (window.mode != "llm_gen") {
        $('#outputs-column').show();
    }

    // Store the current checked states of the split buttons
    const splitCheckedStates = {};
    $('.btn-check-split').each(function () {
        splitCheckedStates[$(this).data('content')] = $(this).is(':checked');
    });

    // Store the current checked states of the output buttons
    const outputCheckedStates = {};
    $('.btn-check-output').each(function () {
        outputCheckedStates[$(this).data('content')] = $(this).is(':checked');
    });

    $('#splits-container').empty();
    $('#outputs-container').empty();

    // unique splits
    const splits = available_data
        .filter(model_out => selectedDatasets.includes(model_out.dataset))
        .map(model_out => model_out.split)
        .filter((value, index, self) => self.indexOf(value) === index);

    for (const split of splits) {
        addSplitButton(split);
    }

    // unique outputs
    const outputs = available_data
        .filter(model_out => selectedDatasets.includes(model_out.dataset) && selectedSplits.includes(model_out.split))
        .map(model_out => model_out.setup_id)
        .filter((value, index, self) => self.indexOf(value) === index);

    for (const output of outputs) {
        addOutputButton(output);
    }

    // Sort all the checkboxes alphabetically
    sortCheckboxes($('#splits-container'));
    sortCheckboxes($('#outputs-container'));

    // Restore the checked states of the split buttons
    $('.btn-check-split').each(function () {
        const split = $(this).data('content');
        if (splitCheckedStates[split]) {
            $(this).prop('checked', true);
        }
    });
    // Restore the checked states of the output buttons
    $('.btn-check-output').each(function () {
        const output = $(this).data('content');
        if (outputCheckedStates[output]) {
            $(this).prop('checked', true);
        }
    });
}


function updateSelectedDatasets() {
    var selectedData = gatherComparisonData();

    if (mode == 'llm_eval' || mode == 'crowdsourcing') {
        $("#selectedDatasetsContent").html(
            selectedData.map(d =>
                `<tr>
                <td>${d.dataset}</td>
                <td>${d.split}</td>
                <td>${d.setup_id}</td>
                <td>${d.output_ids.length}</td>
                <td><button type="button" class="btn btn-sm btn-secondary" onclick="deleteRow(this);">x</button></td>
            </tr>`
            ).join("\n")
        );
    } else if (mode == 'llm_gen') {
        $("#selectedDatasetsContent").html(
            selectedData.map(d =>
                `<tr>
                <td>${d.dataset}</td>
                <td>${d.split}</td>
                <td>${d.output_ids.length}</td>
                <td><button type="button" class="btn btn-sm btn-secondary" onclick="deleteRow(this);">x</button></td>
            </tr>`
            ).join("\n")
        );
    }
}


$(document).on('change', '.btn-check-dataset', updateComparisonData);
$(document).on('change', '.btn-check-split', updateComparisonData);

$(document).on('change', "#data-select-area input[type='checkbox']", function () {
    updateSelectedDatasets();
});

$(document).ready(function () {
    populateDatasets();
});