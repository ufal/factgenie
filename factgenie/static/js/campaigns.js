const available_data = window.available_data;

function clearCampaign(campaignId) {
    // ask for confirmation
    if (!confirm("Are you sure you want to clear all campaign outputs?")) {
        return;
    }
    $.post({
        url: `${url_prefix}/clear_campaign`,
        contentType: 'application/json',
        data: JSON.stringify({
            campaignId: campaignId,
            mode: mode
        }),
        success: function (response) {
            console.log(response);
            window.location.reload();
        }
    });
}

function clearOutput(campaignId, mode, idx) {
    // ask for confirmation
    if (!confirm(`Are you sure you want to free the batch id ${idx}? All related outputs will be deleted.`)) {
        return;
    }
    $.post({
        url: `${url_prefix}/clear_output`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            campaignId: campaignId,
            mode: mode,
            idx: idx,
        }),
        success: function (response) {
            console.log(response);

            if (response.success !== true) {
                alert(response.error);
            } else {
                // reload the page
                location.reload();
            }
        }
    });
}

function createLLMCampaign() {
    const campaignId = $('#campaignId').val();
    // const llmConfig = $('#llmConfig').val();

    const config = gatherConfig();
    var campaignData = gatherSelectedCombinations();

    // if no annotation categories are created, show an alert
    if (mode != "llm_gen" && config.annotationSpanCategories.length == 0) {
        alert("Please add at least one annotation span category.");
        return;
    }

    // if no datasets are selected, show an alert
    if (campaignData.length == 0) {
        alert("Please select at least one existing combination of dataset, split, and output.");
        return;
    }

    $.post({
        url: `${url_prefix}/${mode}/create`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            campaignId: campaignId,
            campaignData: campaignData,
            config: config
        }),
        success: function (response) {
            console.log(response);

            if (response.success !== true) {
                alert(response.error);
            } else {
                window.location.href = `${url_prefix}/${mode}`;
            }
        }
    });
}

function createHumanCampaign() {
    const campaignId = $('#campaignId').val();
    const config = gatherConfig();
    var campaignData = gatherSelectedCombinations();

    // if no datasets are selected, show an alert
    if (campaignData.length == 0) {
        alert("Please select at least one existing combination of dataset, split, and output.");
        return;
    }

    $.post({
        url: `${url_prefix}/crowdsourcing/create`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            campaignId: campaignId,
            config: config,
            campaignData: campaignData
        }),
        success: function (response) {
            console.log(response);

            if (response.success !== true) {
                alert(response.error);
            } else {
                window.location.href = `${url_prefix}/crowdsourcing`;
            }
        }
    });
}

function deleteCampaign(campaignId, mode) {
    // ask for confirmation
    if (!confirm(`Are you sure you want to delete the campaign ${campaignId}? All the data will be lost!`)) {
        return;
    }

    $.post({
        url: `${url_prefix}/delete_campaign`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            campaignId: campaignId,
            mode: mode,
        }),
        success: function (response) {
            console.log(response);

            if (response.success !== true) {
                alert(response.error);
            } else {
                // remove the campaign from the list
                $(`#campaign-${campaignId}`).remove();

                // reload the page
                location.reload();
            }
        }
    });
}

function duplicateConfig(btnElem, filenameElemId, modeTo, campaignId) {
    const filename = $("#" + filenameElemId).val() + ".yaml";
    const modeFrom = window.mode;

    // TODO warn overwrite
    $.post({
        url: `${url_prefix}/duplicate_config`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            campaignId: campaignId,
            modeFrom: modeFrom,
            modeTo: modeTo,
            filename: filename,
        }),
        success: function (response) {
            console.log(response);

            if (response.success !== true) {
                alert(response.error);
            } else {
                // change color of the button save-cfg-submit to green for a second with the label "Saved!", then back to normal
                const origText = $(btnElem).text();
                $(btnElem).removeClass("btn-primary").addClass("btn-success").text("Saved!");
                setTimeout(function () {
                    $('#save-cfg-modal').modal('hide');
                    $(btnElem).removeClass("btn-success").addClass("btn-primary").text(origText);
                }, 1500);
            }
        }
    });
}

function duplicateEval(inputDuplicateId, campaignId) {
    newCampaignId = $(`#${inputDuplicateId}`).val();

    $.post({
        url: `${url_prefix}/duplicate_eval`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            campaignId: campaignId,
            newCampaignId: newCampaignId,
            mode: window.mode
        }),
        success: function (response) {
            console.log(response);

            if (response.success !== true) {
                alert(response.error);
            } else {
                // hide the modal and reload the page
                $('#duplicate-eval-modal').modal('hide');
                location.reload();
            }
        }
    });
}


function gatherConfig() {
    var config = {};

    if (window.mode == "crowdsourcing") {
        config.annotatorInstructions = annotatorInstructionsMDE.value();
        // config.annotatorPrompt = $("#annotatorPrompt").val();
        config.finalMessage = finalMessageMDE.value();
        config.examplesPerBatch = $("#examplesPerBatch").val();
        config.annotatorsPerExample = $("#annotatorsPerExample").val();
        config.idleTime = $("#idleTime").val();
        config.annotationGranularity = $("#annotationGranularity").val();
        config.annotationOverlapAllowed = $("#annotationOverlapAllowed").is(":checked");
        config.service = $("#service").val();
        config.sortOrder = $("#sortOrder").val();
        config.annotationSpanCategories = getAnnotationSpanCategories();
        config.flags = getKeys($("#flags"));
        config.options = getOptions();
        config.sliders = getSliders();
        config.textFields = getKeys($("#textFields"));
    } else if (window.mode == "llm_eval" || window.mode == "llm_gen") {
        config.promptStrat = $("#prompt-strat").val();
        config.modelName = $("#model-name").val();
        config.apiProvider = $("#api-provider").val();
        config.promptTemplate = $("#prompt-template").val();
        config.systemMessage = $("#system-message").val();
        config.annotationOverlapAllowed = $("#annotationOverlapAllowed").is(":checked");
        config.apiUrl = $("#api-url").val();
        config.modelArguments = getKeysAndValues($("#model-arguments"));
        config.extraArguments = getKeysAndValues($("#extra-arguments"));

        if (window.mode == "llm_eval") {
            config.annotationSpanCategories = getAnnotationSpanCategories();
            config.purpose = "metric"
        }
        if (window.mode == "llm_gen") {
            config.startWith = $("#start-with").val();
            config.purpose = "gen"
        }
    }
    return config;
}


function pauseLLMCampaign(campaignId) {
    $(`#run-button-${campaignId}`).show();
    $(`#stop-button-${campaignId}`).hide();
    setCampaignStatus(campaignId, "idle");

    $.post({
        url: `${url_prefix}/llm_campaign/pause`,
        contentType: 'application/json',
        data: JSON.stringify({
            campaignId: campaignId
        }),
        success: function (response) {
            console.log(response);
        }
    });
}

function prefillInstructions() {
    const annotationSpanCategories = getAnnotationSpanCategories();
    const defaultInstructions = window.default_prompts.crowdsourcing;

    if (annotationSpanCategories.length == 0) {
        alert("Please add at least one annotation span category.");
        return;
    }
    // if annotatorInstructionsMDE contains some text, ask for confirmation
    if (annotatorInstructionsMDE.value().length > 0) {
        if (!confirm("Are you sure you want to overwrite the current instructions?")) {
            return;
        }
    }

    var errorList = [];
    annotationSpanCategories.forEach((category) => {
        const span = `- <span style="color: ${category.color}; text-decoration: underline; text-decoration-thickness: 4px; text-decoration-skip-ink: none"><b>${category.name}</b></span>: ${category.description}`;
        errorList.push(span);
    });
    var instructions = defaultInstructions.replace(/{error_list}/g, errorList.join("\n") + "\n");
    annotatorInstructionsMDE.value(instructions);
}

function prefillPrompt() {
    const annotationSpanCategories = getAnnotationSpanCategories();
    const defaultPrompt = window.default_prompts.llm_eval;

    if (annotationSpanCategories.length == 0) {
        alert("Please add at least one annotation span category.");
        return;
    }

    // if promptTemplate contains some text, ask for confirmation
    if ($("#prompt-template").val().length > 0) {
        if (!confirm("Are you sure you want to overwrite the current prompt?")) {
            return;
        }
    }
    var errorList = [];

    annotationSpanCategories.forEach((category, idx) => {
        const span = `${idx}: ${category.name} (${category.description})`;
        errorList.push(span);
    });

    var prompt = defaultPrompt.replace(/{error_list}/g, errorList.join("\n"));
    $("#prompt-template").val(prompt);
}


function runLLMCampaign(campaignId) {
    $(`#run-button-${campaignId}`).hide();
    $(`#stop-button-${campaignId}`).show();
    setCampaignStatus(campaignId, "running");

    startLLMCampaignListener(campaignId);

    $.post({
        url: `${url_prefix}/${mode}/run`,
        contentType: 'application/json',
        data: JSON.stringify({
            campaignId: campaignId
        }),
        success: function (response) {
            if (response.success !== true) {
                alert(response.error);
                $("#log-area").text(JSON.stringify(response.error));
                console.log(JSON.stringify(response));

                setCampaignStatus(campaignId, "idle");
                $(`#run-button-${campaignId}`).show();
                $(`#stop-button-${campaignId}`).hide();
            } else {
                console.log(response);
            }
        }
    });
}

function saveConfig(mode) {
    const filename = $("#config-save-filename").val() + ".yaml";
    const config = gatherConfig();

    if (filename in window.configs) {
        if (!confirm(`The configuration with the name ${filename} already exists. Do you want to overwrite it?`)) {
            return;
        }
    }
    $.post({
        url: `${url_prefix}/save_config`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            mode: mode,
            filename: filename,
            config: config
        }),
        success: function (response) {
            console.log(response);

            if (response.success !== true) {
                alert(response.error);
            } else {
                // change color of the button save-cfg-submit to green for a second with the label "Saved!", then back to normal
                $("#save-cfg-submit").removeClass("btn-primary").addClass("btn-success").text("Saved!");
                setTimeout(function () {
                    $('#save-cfg-modal').modal('hide');
                    $("#save-cfg-submit").removeClass("btn-success").addClass("btn-primary").text("Save");
                }, 1500);
            }
        }
    });

}

function saveGenerationOutputs(campaignId) {
    modelName = $("#save-generations-model-name").val();

    $.post({
        url: `${url_prefix}/save_generation_outputs`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            campaignId: campaignId,
            modelName: modelName
        }),
        success: function (response) {
            console.log(response);

            if (response.success !== true) {
                alert(response.error);
            } else {
                // change color of the button save-cfg-submit to green for a second with the label "Saved!", then back to normal
                $("#save-generations-submit").removeClass("btn-primary").addClass("btn-success").text("Saved!");
                setTimeout(function () {
                    $('#save-generations-modal').modal('hide');
                    $("#save-generations-submit").removeClass("btn-success").addClass("btn-primary").text("Save");
                }, 1500);
            }
        }
    });
}

function setCampaignStatus(campaignId, status) {
    $(`#metadata-status-${campaignId}`).html(status);
    $(`#metadata-status-${campaignId}`).removeClass("bg-idle bg-running bg-finished bg-error");
    $(`#metadata-status-${campaignId}`).addClass(`bg-${status}`);
}

function setExampleStatus(status, button) {
    button.removeClass("bg-free bg-finished");
    button.addClass(`bg-${status}`);
    button.text(status);
}


function showResult(payload, campaignId) {
    const finished_examples = payload.stats.finished;
    const total_examples = payload.stats.total;
    const progress = Math.round((finished_examples / total_examples) * 100);
    $(`#llm-progress-bar-${campaignId}`).css("width", `${progress}%`);
    $(`#llm-progress-bar-${campaignId}`).attr("aria-valuenow", progress);
    $(`#metadata-example-cnt-${campaignId}`).html(`${finished_examples} / ${total_examples}`);
    console.log(`Progress: ${progress}%`);


    // update the annotation button
    const example = payload.response;
    const dataset = example.dataset;
    const split = example.split;
    const setup_id = example.setup_id || campaignId;
    const example_idx = example.example_idx;
    const rowId = `${dataset}-${split}-${setup_id}-${example_idx}`;
    const annotation_button = $(`#annotBtn${rowId}`);
    annotation_button.show();

    const clear_output_button = $(`#clearOutput${rowId}`);
    clear_output_button.show();

    // update the annotation content
    const annotation_div = $(`#annotPre${rowId}`);

    // llm_eval mode
    if (example.annotations !== undefined) {
        annotation_div.text(JSON.stringify(example.annotations));
    } else { // llm_gen mode
        annotation_div.text(example.output);
    }
    $(`#annotCard${rowId}`).show();
    // update the status
    const status_button = $(`#statusBtn${rowId}`);
    setExampleStatus("finished", status_button);
}

function finalizeCampaign(campaignId) {
    console.log("Closing the connection");

    setCampaignStatus(campaignId, "finished");
    $(`#run-button-${campaignId}`).hide();
    $(`#stop-button-${campaignId}`).hide();
    $(`#download-button-${campaignId}`).show();

    if (window.mode == "llm_gen") {
        $("#save-generations-button").show();
    }

}

function startLLMCampaignListener(campaignId) {
    var source = new EventSource(`${url_prefix}/llm_campaign/progress/${campaignId}`);
    console.log(`Listening for progress events for campaign ${campaignId}`);

    source.onmessage = function (event) {
        // update the progress bar
        var payload = JSON.parse(event.data);

        if (payload.type === "status") {
            console.log(payload.message);
        }
        else if (payload.type === "result") {
            showResult(payload, campaignId);

            if (payload.stats.finished == payload.stats.total) {
                source.close();
                finalizeCampaign(campaignId);
            }
        }
    };
}


function updateCampaignConfig(campaignId) {
    // collect values of all .campaign-metadata textareas, for each input also extract the key in `data-key`
    var config = {};
    $(`.campaign-metadata-${campaignId}`).each(function () {
        const key = $(this).data("key");
        const value = $(this).val();
        config[key] = value;
    });

    $.post({
        url: `${url_prefix}/llm_campaign/update_metadata`,
        contentType: 'application/json',
        data: JSON.stringify({
            campaignId: campaignId,
            config: config
        }),
        success: function (response) {
            console.log(response);
            $(".update-config-btn").removeClass("btn-danger").addClass("btn-success").text("Saved!");

            setTimeout(function () {
                $(`#config-modal-${campaignId}`).modal('hide');
                $(".update-config-btn").removeClass("btn-success").addClass("btn-danger").text("Update configuration");
            }, 1500);

        }
    });
}

function updateCrowdsourcingConfig() {
    const crowdsourcingConfig = $('#crowdsourcingConfig').val();

    if (crowdsourcingConfig === "[None]") {
        annotatorInstructionsMDE.value("");
        // $("#annotatorPrompt").val("");
        finalMessageMDE.value("");
        $("#examplesPerBatch").val("");
        $("#annotatorsPerExample").val("");
        $("#idleTime").val("");
        $("#annotation-span-categories").empty();
        $("#flags").empty();
        $("#options").empty();
        $("#sliders").empty();
        $("#textFields").empty();
        return;
    }
    const cfg = window.configs[crowdsourcingConfig];

    const annotatorInstructions = cfg.annotator_instructions;
    const finalMessage = cfg.final_message;
    const examplesPerBatch = cfg.examples_per_batch;
    const annotatorsPerExample = cfg.annotators_per_example;
    const idleTime = cfg.idle_time;
    const annotationGranularity = cfg.annotation_granularity;
    const annotationOverlapAllowed = cfg.annotation_overlap_allowed;
    const service = cfg.service;
    const sortOrder = cfg.sort_order;
    const annotationSpanCategories = cfg.annotation_span_categories;
    const flags = cfg.flags;
    const options = cfg.options;
    const sliders = cfg.sliders;
    const textFields = cfg.text_fields;

    annotatorInstructionsMDE.value(annotatorInstructions);
    // $("#annotatorPrompt").val(annotatorPrompt);
    finalMessageMDE.value(finalMessage);
    $("#examplesPerBatch").val(examplesPerBatch);
    $("#annotatorsPerExample").val(annotatorsPerExample);
    $("#idleTime").val(idleTime);
    $("#annotationGranularity").val(annotationGranularity);
    $("#annotationOverlapAllowed").prop("checked", annotationOverlapAllowed);
    $("#service").val(service);
    $("#sortOrder").val(sortOrder);
    $("#annotation-span-categories").empty();

    annotationSpanCategories.forEach((annotationSpanCategory) => {
        addAnnotationSpanCategory(annotationSpanCategory.name, annotationSpanCategory.description, annotationSpanCategory.color);
    });
    $("#flags").empty();

    if (flags !== undefined) {
        flags.forEach((flag) => {
            const newFlag = createFlagElem(flag);
            $("#flags").append(newFlag);
        });
    }

    $("#options").empty();

    if (options !== undefined) {
        options.forEach((option) => {
            const newOption = createOptionElem(option.label, option.values.join(", "));
            $("#options").append(newOption);
        });
    }

    $("#sliders").empty();

    if (sliders !== undefined) {
        sliders.forEach((slider) => {
            const newSlider = createSliderElem(slider.label, slider.min, slider.max, slider.step);
            $("#sliders").append(newSlider);
        });
    }

    $("#textFields").empty();

    if (textFields !== undefined) {
        textFields.forEach((textField) => {
            const newTextField = createTextFieldElem(textField);
            $("#textFields").append(newTextField);
        });
    }
}



function updateLLMMetricConfig() {
    const llmConfigValue = $('#llmConfig').val();

    if (llmConfigValue === "[None]") {
        $("#model-name").val("");
        $("#prompt-template").val("");
        $("#system-message").val("");
        $("#api-url").val("");
        $("#model-arguments").empty();
        $("#annotation-span-categories").empty();
        $("#extra-arguments").empty();
        $("#annotationOverlapAllowed").prop("checked", false);
        return;
    }
    const cfg = window.configs[llmConfigValue];

    // Supporting a deprecated `type` field
    const api_provider = cfg.api_provider || cfg.type;
    const prompt_strat = cfg.prompt_strat;
    const model_name = cfg.model;
    const prompt_template = cfg.prompt_template;
    const system_msg = cfg.system_msg;
    const annotation_overlap_allowed = cfg.annotation_overlap_allowed;
    const api_url = cfg.api_url;
    const model_args = cfg.model_args;
    const extra_args = cfg.extra_args;

    // for metric, we need to select the appropriate one from the values in the select box
    $("#api-provider").val(api_provider);
    $("#prompt-strat").val(prompt_strat);
    $("#model-name").html(model_name);
    $("#prompt-template").html(prompt_template);
    $("#system-message").html(system_msg);
    $("#annotationOverlapAllowed").prop("checked", annotation_overlap_allowed);
    $("#api-url").html(api_url);
    $("#model-arguments").empty();
    $("#extra-arguments").empty();

    $.each(model_args, function (key, value) {
        const newArg = createArgElem(key, value);
        $("#model-arguments").append(newArg);
    });

    $.each(extra_args, function (key, value) {
        const newArg = createArgElem(key, value);
        $("#extra-arguments").append(newArg);
    });

    if (mode == "llm_eval") {
        const annotationSpanCategories = cfg.annotation_span_categories;
        $("#annotation-span-categories").empty();

        annotationSpanCategories.forEach((annotationSpanCategory) => {
            addAnnotationSpanCategory(annotationSpanCategory.name, annotationSpanCategory.description, annotationSpanCategory.color);
        });
    }
    if (mode == "llm_gen") {
        const start_with = cfg.start_with;
        $("#start-with").val(start_with);
    }
}
