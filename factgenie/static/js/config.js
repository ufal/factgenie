
function updateConfig() {
    const config = {
        logging: {
            flask_debug: $('#debug').is(':checked'),
            level: $('#logging_level').val(),
        },
        host_prefix: $('#host_prefix').val(),
        login: {
            active: $('#login_active').is(':checked'),
            lock_view_pages: $('#lock_view_pages').is(':checked'),
            username: $('#login_username').val(),
            password: $('#login_password').val()
        },
        api_keys: {
            OPENAI_API_KEY: $('#openai_api_key').val(),
            ANTHROPIC_API_KEY: $('#anthropic_api_key').val(),
            GEMINI_API_KEY: $('#gemini_api_key').val(),
            VERTEXAI_PROJECT: $('#vertexai_project').val(),
            VERTEXAI_LOCATION: $('#vertexai_location').val(),
            VERTEXAI_JSON_FULL_PATH: $('#vertexai_json_full_path').val(),
        }
    };

    $.post({
        url: `${url_prefix}/update_config`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify(config),
        success: function (response) {
            alert('Configuration updated successfully');
        },
        error: function (error) {
            alert('Error updating configuration: ' + error.responseText);
        }
    });

}


$(document).ready(function () {
    $("#show_hide_password a").on('click', function (event) {
        event.preventDefault();
        if ($('#show_hide_password input').attr("type") == "text") {
            $('#show_hide_password input').attr('type', 'password');
            $('#show_hide_password i').addClass("fa-eye-slash");
            $('#show_hide_password i').removeClass("fa-eye");
        } else if ($('#show_hide_password input').attr("type") == "password") {
            $('#show_hide_password input').attr('type', 'text');
            $('#show_hide_password i').removeClass("fa-eye-slash");
            $('#show_hide_password i').addClass("fa-eye");
        }
    });
});