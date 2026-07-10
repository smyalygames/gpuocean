export SINGULARITYENV_LD_PRELOAD="/opt/cray/pe/mpich/9.0.1/gtl/lib/libmpi_gtl_hsa.so.0"

export SINGULARITYENV_LD_LIBRARY_PATH=\
"${CRAY_LD_LIBRARY_PATH}:"\
'/lib64:'\
"/opt/cray/pe/mpich/default/ofi/gnu/12.3/lib-abi-mpich:"\
'/opt/cray/libfabric/1.22.0/lib64:'\
'/opt/cray/pe/pmi/6.1.15/lib:'\
'/opt/cray/pals/1.6/lib:'\
'/opt/cray/pe/lib64:'\
'/opt/cray/xpmem/default/lib64:'\
'/usr/lib64:'\
"${LD_LIBRARY_PATH}"


export SINGULARITY_BIND=\
'/appl,'\
'/var/spool/slurmd,'\
'/opt/cray,'\
'/var/spool,'\
'/etc/host.conf,'\
'/etc/hosts,'\
'/etc/nsswitch.conf,'\
'/etc/resolv.conf,'\
'/etc/ssl/openssl.cnf,'\
'/usr/lib64/libatomic.so.1,'\
'/usr/lib64/libbrotlicommon.so.1,'\
'/usr/lib64/libbrotlidec.so.1,'\
'/usr/lib64/libcrypto.so.1.1,'\
'/usr/lib64/libcurl.so.4,'\
'/usr/lib64/libcxi.so.1,'\
'/usr/lib64/libdrm_amdgpu.so.1,'\
'/usr/lib64/libdrm.so.2,'\
'/usr/lib64/libelf.so.1,'\
'/opt/cray/pe/gcc-libs/libgcc_s.so.1:/usr/lib64/libgcc_s.so.1,'\
'/opt/cray/pe/gcc-libs/libgfortran.so.5:/usr/lib64/libgfortran.so.5,'\
'/usr/lib64/libgssapi_krb5.so.2,'\
'/usr/lib64/libidn2.so.0,'\
'/usr/lib64/libjansson.so.4,'\
'/usr/lib64/libjitterentropy.so.3,'\
'/usr/lib64/libjson-c.so.5.2.0:/usr/lib64/libjson-c.so.3,'\
'/usr/lib64/libk5crypto.so.3,'\
'/usr/lib64/libkeyutils.so.1,'\
'/usr/lib64/libkrb5.so.3,'\
'/usr/lib64/libkrb5support.so.0,'\
'/usr/lib64/liblber-2.4.so.2,'\
'/usr/lib64/libldap_r-2.4.so.2,'\
'/usr/lib64/liblnetconfig.so.4,'\
'/usr/lib64/liblustreapi.so:/usr/lib/x86_64-linux-gnu/liblustreapi.so,'\
'/usr/lib64/libnghttp2.so.14,'\
'/usr/lib64/libnl-3.so.200,'\
'/usr/lib64/libnl-genl-3.so.200,'\
'/usr/lib64/libnl-route-3.so.200,'\
'/usr/lib64/libnuma.so.1,'\
'/usr/lib64/libpcre.so.1,'\
'/usr/lib64/libpsl.so.5,'\
'/usr/lib64/libsasl2.so.3,'\
'/usr/lib64/libssh.so.4,'\
'/usr/lib64/libssl.so.1.1,'\
'/usr/lib64/libunistring.so.2,'\
'/usr/lib64/libyaml-0.so.2.0.5:/usr/lib64/libyaml-0.so.2,'\
'/usr/lib64/libz.so.1,'\
'/usr/lib64/libzstd.so.1,'\
'/run/cxi,'\
"${EBROOTAWSMINOFIMINNCCL},"
