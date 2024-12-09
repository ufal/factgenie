const modelOutputs = window.model_outputs;

function createUploadDatasetSplitElem() {
    const newSplit = $(`
        <div class="row mt-1">
        <div class="col-4">
        <input type="text" class="form-control split-name" name="split-name" placeholder="Split name">
        </div>
        <div class="col-7">
        <input class="form-control split-file" type="file" name="split-file" required>
        </div>
        <div class="col-1">
        <button type="button" class="btn btn-danger" onclick="deleteRow(this);">x</button>
        </div>
        </div>
    `);
    return newSplit;
}

function deleteRow(button) {
    $(button).parent().parent().remove();
}

function addDatasetSplit() {
    const datasetSplits = $("#dataset-files");
    const newSplit = createUploadDatasetSplitElem();
    datasetSplits.append(newSplit);
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
                window.location.hash = "#annotations";
                // reload the page
                location.reload();
            }
        }
    });
}

function deleteOutput(dataset, split, setup_id) {
    // ask for confirmation
    if (!confirm(`Are you sure you want to delete the output for ${setup_id}?`)) {
        return;
    }
    $.post({
        url: `${url_prefix}/delete_model_outputs`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            dataset: dataset,
            split: split,
            setup_id: setup_id
        }),
        success: function (response) {
            console.log(response);

            if (response.success !== true) {
                alert(response.error);
            } else {
                // reload
                window.location.hash = "#outputs";
                location.reload();
            }
        }
    });
}

function changeDataset() {
    const dataset = $('#dataset-select').val();

    // set available splits in #split-select
    $('#split-select').empty();

    for (const split of datasets[dataset].splits) {
        $('#split-select').append(`<option value="${split}">${split}</option>`);
    }
}


function downloadDataset(datasetId) {
    $(`#btn-download-${datasetId}`).hide();
    // add a loading spinner
    $(`#row-actions-${datasetId}`).append(`
        <div class="spinner-border text-secondary" role="status" id="spinner-download-${datasetId}">
            <span class="visually-hidden">Loading...</span>
        </div>
    `);

    $.post({
        url: `${url_prefix}/download_dataset`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            datasetId: datasetId,
        }),
        success: function (response) {
            console.log(response);

            if (response.success !== true) {
                alert(response.error);
                $(`#spinner-download-${datasetId}`).remove();
                $(`#btn-download-${datasetId}`).show();
            } else {
                // remove the spinner
                // $(`#spinner-download-${datasetId}`).remove();
                // $(`#check-downloaded-${datasetId}`).show();

                window.location.hash = "#local";
                location.reload();
            }
        }
    });

}

function deleteDataset(datasetId) {
    // ask for confirmation
    if (!confirm(`Are you sure you want to delete the dataset ${datasetId}? All the data will be lost!`)) {
        return;
    }

    $.post({
        url: `${url_prefix}/delete_dataset`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            datasetId: datasetId,
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

function uploadModelOutputs() {
    const dataset = $("#dataset-select").val();
    const split = $("#split-select").val();
    const setup_id = $("#setup-id").val();

    if (setup_id === "") {
        alert("Please enter the setup id");
        return;
    }

    // read the file from #model-output-upload form
    const file = $("#model-output-upload")[0].files[0];
    const reader = new FileReader();
    reader.readAsText(file);

    // send by post request to /upload_model_outputs
    reader.onload = function (e) {
        $.post({
            url: `${url_prefix}/upload_model_outputs`,
            contentType: 'application/json', // Specify JSON content type
            data: JSON.stringify({
                outputs: e.target.result,
                dataset: dataset,
                split: split,
                setup_id: setup_id,
            }),
            success: function (response) {
                if (response.success !== true) {
                    alert(response.error);
                } else {
                    window.location.hash = "#outputs";
                    location.reload();
                }
            }
        });
    }
}


function setDatasetEnabled(name, enabled) {
    $.post({
        url: `${url_prefix}/set_dataset_enabled`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            datasetId: name,
            enabled: enabled
        }),
        success: function (response) {
            console.log(response);

            if (response.success !== true) {
                alert(response.error);
            } else {
                // reload
                location.reload();
            }
        }
    });
}



function uploadDataset() {
    const dataset_id = $("#dataset-id").val();
    const description = $("#dataset-description").val();
    const format = $("#dataset-format").val();
    const dataset = {};
    var filesToRead = $("#dataset-files").children().length;

    $("#upload-dataset-btn").text("Uploading...").prop("disabled", true);

    // Function to send the POST request
    function sendRequest() {
        $.post({
            url: `${url_prefix}/upload_dataset`,
            contentType: 'application/json', // Specify JSON content type
            data: JSON.stringify({
                name: dataset_id,
                description: description,
                format: format,
                dataset: dataset,
            }),
            success: function (response) {
                console.log(response);

                if (response.success !== true) {
                    alert(response.error);
                    $("#upload-dataset-btn").text("Upload dataset").prop("disabled", false);
                } else {
                    // reload
                    window.location.hash = "#local";
                    location.reload();
                }
            }
        });
    }

    // Read each file
    $("#dataset-files").children().each(function () {
        const splitName = $(this).find("input[name='split-name']").val();
        const splitFile = $(this).find("input[name='split-file']")[0].files[0];
        const reader = new FileReader();

        reader.onload = function (e) {
            // Check if the file is a ZIP file
            if (splitFile.type === "application/zip") {
                dataset[splitName] = Array.from(new Uint8Array(e.target.result));
            } else {
                dataset[splitName] = e.target.result;
            }
            filesToRead--;

            // If all files are read, send the request
            if (filesToRead === 0) {
                sendRequest();
            }
        };

        // Read as ArrayBuffer for binary files (e.g., ZIP)
        if (splitFile.type === "application/zip") {
            reader.readAsArrayBuffer(splitFile);
        } else {
            reader.readAsText(splitFile);
        }
    });
}



$(document).ready(function () {
    if (Object.keys(datasets).length > 0) {
        $("#dataset-select").val(Object.keys(datasets)[0]).trigger("change");
    }
    // Function to activate the tab based on the anchor
    function activateTabFromAnchor() {
        var anchor = window.location.hash.substring(1);
        if (anchor) {
            var tabToActivate = $('a[data-anchor="' + anchor + '"]');
            if (tabToActivate.length) {
                tabToActivate.tab('show');
            }
        }
    }

    // Add click event listener to update the URL
    $('a[data-bs-toggle="pill"]').on('click', function (e) {
        var anchor = $(this).data('anchor');
        if (anchor) {
            window.location.hash = anchor;
        }
    });

    // Activate the tab based on the URL anchor on page load
    activateTabFromAnchor();

    // Re-activate the tab if the URL hash changes
    $(window).on('hashchange', function () {
        activateTabFromAnchor();
    });


    enableTooltips();
});

$("#dataset-select").on("change", changeDataset);