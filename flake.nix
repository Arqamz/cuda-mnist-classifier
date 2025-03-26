{
  description = "Dev shell for C/C++";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ...}: let
    pkgs = import nixpkgs {
	system = "x86_64-linux";
	config.allowUnfree = true;
    };
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
		
	  packages = with pkgs; [
	    cmake
	    libgcc
	    llvmPackages_19.libcxx
	    libpkgconf
		linuxKernel.packages.linux_6_12.perf
		gprof2dot

	    cudaPackages.cuda_nvcc
	    cudaPackages.cuda_gdb
	    cudaPackages.cutensor
	    cudaPackages.libcublas
	    cudaPackages.libcurand

	  ];
	
	  shellHook = ''
	    export CXX=g++
	    export CC=gcc
	    echo "Dev shell for C/C++/Cuda is ready!"
	  '';
    };
  };
}
