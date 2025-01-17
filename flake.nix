{
  description =
    "A simple python implementation of 2048 and an AI that learns to play it.";

  inputs = { nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable"; };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = { allowUnfree = true; };
      };
    in {
      devShells.x86_64-linux.default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          python312Packages.python
          python312Packages.pip
          python312Packages.tkinter
          python312Packages.torch
          python312Packages.matplotlib
          python312Packages.numpy
          python312Packages.ipython
        ];
        shellHook =
          "  export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/\n  export CUDA_PATH=${pkgs.cudatoolkit}\n";
      };
    };
}

