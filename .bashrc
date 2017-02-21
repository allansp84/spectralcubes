PREFIX_INSTALL=/home/ubuntu/local

if [ -d $PREFIX_INSTALL ]; then
    for i in $PREFIX_INSTALL/* $PREFIX_INSTALL/x86_64/*; do
        [ -d $i/bin ] && PATH=${i}/bin:${PATH}
        [ -d $i/sbin ] && PATH=${i}/sbin:${PATH}
        [ -d $i/include ] && CPATH=${i}/include:${CPATH}
        [ -d $i/include ] && INCLUDE=${i}/include:${INCLUDE}
        [ -d $i/lib ] && LIBDIR=${i}/lib:${LIBDIR}
        [ -d $i/lib ] && LD_RUN_PATH=${i}/lib:${LD_RUN_PATH}
        [ -d $i/lib ] && LD_LIBRARY_PATH=${i}/lib:${LD_LIBRARY_PATH}
        [ -d $i/lib64 ] && LD_LIBRARY_PATH=${i}/lib:${LD_LIBRARY_PATH}
        [ -d $i/libexec ] && LD_LIBRARY_PATH=${i}/lib:${LD_LIBRARY_PATH}
        [ -d $i/lib ] && DYLD_LIBRARY_PATH=${i}/lib:${DYLD_LIBRARY_PATH}
        [ -d $i/lib/pkgconfig ] && PKG_CONFIG_PATH=${i}/lib/pkgconfig:${PKG_CONFIG_PATH}
        [ -d $i/share/man ] && MANPATH=${i}/share/man:${MANPATH}
    done
    export PATH
    export CPATH
    export INCLUDE
    export LIBDIR
    export LD_RUN_PATH
    export LD_LIBRARY_PATH
    export PKG_CONFIG_PATH
    export MANPATH
fi

export PYTHONPATH=$PREFIX_INSTALL/opencv-2.4.13/lib/python2.7/dist-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$PREFIX_INSTALL/opencv-2.4.13/lib:$LD_LIBRARY_PATH
