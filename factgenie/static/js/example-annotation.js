// Initialize an array to store annotations
let annotations = [];

// Function to update the outputarea with the annotations JSON
function updateOutputArea() {
    const output = { "annotations": annotations.map(({ id, ...rest }) => rest) };
    $('#outputarea').text(JSON.stringify(output, null, 4));
}

// Function to handle adding a new annotation
function onAnnotationAdded(annotation) {
    // Extract the necessary information from the annotation
    const categoryIndex = annotation.attributes.type;
    const annotatedText = annotation.attributes.text;
    const annotationId = annotation.cid;

    // Add the annotation to the annotations array
    annotations.push({
        id: annotationId,
        type: categoryIndex,
        text: annotatedText,
        reason: ""
    });

    // Add a new row to the errorarea
    const row = $(`
         <tr id="error-row-${annotationId}">
             <td>${categoryIndex}</td>
             <td>${annotatedText}</td>
             <td>
                 <input type="text" class="form-control reason-input" data-id="${annotationId}" placeholder="Enter reason">
             </td>
         </tr>
     `);
    $('#errorarea').append(row);

    // Attach event listener to the reason input
    row.find('.reason-input').on('input', function () {
        const id = $(this).data('id');
        const reason = $(this).val();
        // Update the reason in the annotations array
        const annotation = annotations.find(a => a.id === id);
        if (annotation) {
            annotation.reason = reason;
            updateOutputArea();
        }
    });

    // Update the output area
    updateOutputArea();
}

// Function to handle deleting an annotation
function onAnnotationDeleted(annotation) {
    const annotationId = annotation.cid;

    // Remove the annotation from the annotations array
    annotations = annotations.filter(a => a.id !== annotationId);

    // Remove the corresponding row from errorarea
    $(`#error-row-${annotationId}`).remove();

    // Update the output area
    updateOutputArea();
}

function pasteExampleIntoPrompt() {
    const exampleData = $('#example-data').val();
    const exampleText = $('#example-text').val();
    const output = $('#outputarea').text();
    const prompt = `\n\n*Example:*\ninput:\n\`\`\`\n${exampleData}\n\`\`\`\ntext:\n\`\`\`\n${exampleText}\n\`\`\`\noutput:\n\`\`\`\n${output}\n`;
    $('#prompt-template').val($('#prompt-template').val() + prompt);
    $('#exampleAnnotation').modal('hide');
}

function createButtons() {
    // if buttons already exist, remove them
    $('#buttons-area').empty();

    const annotationSpanCategories = getAnnotationSpanCategories();

    for (const [idx, category] of Object.entries(annotationSpanCategories)) {
        const input = $('<input>', {
            type: "radio",
            class: "btn-check btn-outline-secondary btn-err-cat",
            name: "btnradio",
            id: `btnradio${idx}`,
            autocomplete: "off",
            "data-cat-idx": idx
        });

        const label = $('<label>', {
            class: "btn btn-err-cat-label",
            for: `btnradio${idx}`,
            style: `background-color: ${category.color};`
        }).text(category.name);

        if (idx == 0) {
            input.attr('checked', 'checked');
            label.addClass('active');
        }

        $('#buttons-area').append(input);
        $('#buttons-area').append(label);

        $(".btn-err-cat").change(function () {
            if (this.checked) {
                const cat_idx = $(this).attr("data-cat-idx");
                YPet.setCurrentAnnotationType(cat_idx);
            }
        });

        $('.btn-check').on('change', function () {
            $('.btn-check').each(function () {
                const label = $(`label[for=${this.id}]`);
                if (this.checked) {
                    label.addClass('active');
                } else {
                    label.removeClass('active');
                }
            });
        });
    }
}

function initAnnotation() {
    $("#nextsteparea").show();
    $('#errorarea').empty();
    $('#outputarea').empty();

    console.log("Initializing annotation...");
    const annotationSpanCategories = getAnnotationSpanCategories();

    YPet.addInitializer(function (options) {
        /* Configure the # and colors of Annotation types (minimum 1 required) */
        YPet.AnnotationTypes = new AnnotationTypeList(annotationSpanCategories);

        const exampleText = $('#example-text').val();
        const p = new Paragraph({ 'text': exampleText, 'granularity': "words" });

        // Create a region for the output area
        const regions = { 'annotationarea': '#annotationarea' };
        YPet.addRegions(regions);

        // Show the paragraph in the output area
        YPet.annotationarea.show(new WordCollectionView({ collection: p.get('words') }));

        // Handle annotation removal
        YPet.annotationarea.currentView.collection.parentDocument.get('annotations').on('remove', function (model, collection) {
            if (collection.length == 0) {
                collection = [];
            }
            onAnnotationDeleted(model);
        });

        // Handle annotation addition
        YPet.annotationarea.currentView.collection.parentDocument.get('annotations').on('add', function (model, collection) {
            onAnnotationAdded(model);
        });

    });
    YPet.start();

}

function checkAndOpenModal() {
    const annotationSpanCategories = getAnnotationSpanCategories();
    if (annotationSpanCategories.length == 0) {
        alert("Please add at least one annotation span category.");
        return;
    }
    createButtons();
    var modal = new bootstrap.Modal('#exampleAnnotation');
    modal.show();
}