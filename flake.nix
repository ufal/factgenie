{
  description = "Factgenie devenv flake.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-24.11";
    unstable-nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs, unstable-nixpkgs }:
  let
    system = "x86_64-linux";

    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
      };
    };

    unstable-pkgs = import unstable-nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
      };
    };

    python-packages = pkgs.python311Packages;

    venvDir = ".venv";

    make-pypi = {pname, version, hash}: python-packages.buildPythonPackage {
      inherit pname version;
      src = python-packages.fetchPypi {
        inherit pname version hash;
      };

      nativeBuildInputs = [
        python-packages.flask
        python-packages.redis
        python-packages.six
        python-packages.setuptools
        python-packages.wheel
        python-packages.cmake
        python-packages.apscheduler
      ];

      dontUseCmakeConfigure = true;
      # pyproject = true;
      doCheck = false;
    };

    pypi = {
      flask-sse = make-pypi {
        pname = "Flask-SSE";
        version = "1.0.0";
        hash = "sha256-T4RxTCVJpF5PF7/F9o7oqfKYsidApoREBNHHRVHyCQ0";
      };
      flask-apscheduler = make-pypi {
        pname = "Flask-APScheduler";
        version = "1.13.1";
        hash = "sha256-uSmEbwJvszm3Y2Cw5Px12ni3XG0IYlcVvQ03lJvWB9o=";
      };
      json2table = make-pypi {
        pname = "json2table";
        version = "1.1.5";
        hash = "sha256-g45MBSJ2JS/sew29EGaCWM4Jl2BYiKJQMHvWsSSubQo=";
      };
      tinyhtml = make-pypi {
        pname = "tinyhtml";
        version = "1.2.0";
        hash = "sha256-5+43bYOlUviEmUaI3eNkcQqMsPXlyFhYfnvrCx98Bnk=";
      };
      pygamma-agreement = make-pypi {
        pname = "pygamma-agreement";
        version = "0.5.9";
        hash = "sha256-bZpFk4PhJ55AtM9WTDMld9rQubOT4pGQRKsexTeHnic=";
      };
    };

    python-with-packages = (pkgs.python311.withPackages (p: with p; [
      apscheduler
      coloredlogs
      cvxpy
      datasets
      flask
      litellm
      lxml
      markdown
      natsort
      numba
      openai
      plotly
      pyannote-core
      pyannote-database
      pyannote-metrics
      pyannote-pipeline
      pypi.flask-apscheduler
      pypi.flask-sse
      pypi.json2table
      pypi.pygamma-agreement
      pypi.tinyhtml
      python-slugify
      pyyaml
      requests
      scipy
    ]));

    shellHook = ''
      # Create virtualenv if doesn't exist
      if test ! -d "./${venvDir}"; then
        echo "creating venv..."
        virtualenv "./${venvDir}" --python="${python-with-packages}/bin/python3" --system-site-packages
        export PYTHONHOME="${python-with-packages}"  # This has to be set AFTER creating virtualenv but BEFORE installing requirements. That is because the --system-site-packages tells the virtualenv to use system packages when they are available (such as the tricky numpy package).

        if test -f requirements.txt; then
          echo "installing dependencies... ('pip install -r requirements.txt')"
          ${venvDir}/bin/pip install -r requirements.txt
        fi
        if test -f setup.py || test -f pyproject.toml; then
          echo "installing dependencies... ('pip install -e .')"
          ${venvDir}/bin/pip install -e .
        fi
      fi

      export PYTHONHOME="${python-with-packages}"
      nu --execute "overlay use ${venvDir}/bin/activate.nu; \$env.GEMINI_API_KEY = (gpg -d google-api-key.txt.gpg)"
      exit  # after exiting nushell, also exit the nix shell
    '';

    factgenie = python-packages.buildPythonPackage {
      pname = "factgenie";
      version = "1.1.1";
      src = ./.;
      # Propagated = anything that uses factgenie will also see these
      propagatedBuildInputs = with python-packages; [
        apscheduler
        coloredlogs
        datasets
        flask
        lxml
        markdown
        natsort
        openai
        pypi.flask-apscheduler
        pypi.flask-sse
        pypi.json2table
        pypi.tinyhtml
        python-slugify
        pyyaml
        requests
        scipy
      ];
    };
  in
  {
    devShells.${system}.default = pkgs.mkShellNoCC {
      packages = [
        python-with-packages
        pkgs.virtualenv
        unstable-pkgs.ollama-cuda
      ];

      inherit shellHook;
    };

    apps.${system}.default = {
      type = "app";
      program = "${factgenie}/bin/factgenie";
    };
  };
}
