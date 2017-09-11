! Routines to call tenstream with optical properties from RRTM

module m_tenstr_rrtm_lw

      use mpi, only : mpi_comm_rank

      use m_tenstr_rrtmg_lw_init, only: rrtmg_lw_ini
      use m_tenstr_parkind_lw, only: im => kind_im, rb => kind_rb
      use m_tenstr_rrlw_wvn, only : ngb, wavenum1, wavenum2
      use m_tenstr_parrrtm, only: ngptlw, nbndlw
      use m_tenstr_rrtmg_lw_rad, only: rrtmg_lw

      use m_data_parameters, only : init_mpi_data_parameters, &
        iintegers, ireals, myid, zero, one, i0, i1,           &
        mpiint, pi, mpierr, default_str_len

      use m_tenstream, only : init_tenstream, set_optical_properties, solve_tenstream, destroy_tenstream,&
          tenstream_get_result, tenstream_get_result_toZero, C_one, C_one1

      use m_helper_functions, only : read_ascii_file_2d, gradient, meanvec, imp_bcast, &
          imp_allreduce_min, imp_allreduce_max, search_sorted_bisection, CHKERR
      use m_tenstream_interpolation, only : interp_1d

      use m_netcdfIO, only : ncwrite

  implicit none

  private
  public :: tenstream_rrtm_lw, destroy_tenstream_rrtm_lw

  integer(mpiint) :: ierr

  logical :: linit_tenstr=.False.

  logical,parameter :: ldebug=.True.


  interface
    real function PLKINT(WVLLO, WVLHI, T)
      real :: WVLLO, WVLHI, T
    end function 
  end interface

  type t_atm
    real(ireals),allocatable :: plev   (:) ! dim(nlay+1)
    real(ireals),allocatable :: tlev   (:) !
    real(ireals),allocatable :: zt     (:) !
    real(ireals),allocatable :: h2o_lev(:) !
    real(ireals),allocatable :: o3_lev (:) !
    real(ireals),allocatable :: co2_lev(:) !
    real(ireals),allocatable :: ch4_lev(:) !
    real(ireals),allocatable :: n2o_lev(:) !
    real(ireals),allocatable :: o2_lev (:) !

    real(ireals),allocatable :: play   (:) ! dim(nlay)
    real(ireals),allocatable :: zm     (:) !
    real(ireals),allocatable :: dz     (:) !
    real(ireals),allocatable :: tlay   (:) !
    real(ireals),allocatable :: h2o_lay(:) !
    real(ireals),allocatable :: o3_lay (:) !
    real(ireals),allocatable :: co2_lay(:) !
    real(ireals),allocatable :: ch4_lay(:) !
    real(ireals),allocatable :: n2o_lay(:) !
    real(ireals),allocatable :: o2_lay (:) !
  end type
  type(t_atm),allocatable :: bg_atm

contains
  subroutine sanitize_input(plev, tlev, tlay)
    real(ireals),intent(in),dimension(:) :: plev, tlev
    real(ireals),intent(in),dimension(:),optional :: tlay

<<<<<<< HEAD
    integer(iintegers) ::  errcnt
    logical :: ierr
 
    errcnt = 0
    ierr = maxval(plev) .gt. 1050; 
    if(ierr) then 
        print *,'Pressure above 1050 hPa -- are you sure this is earth?', maxval(plev)
        errcnt = errcnt + 1 
    endif

    ierr = minval(plev) .lt. zero; 
    if(ierr) then  
        print *,'Pressure negative -- are you sure this is physically correct?', minval(plev)
        errcnt = errcnt + 1 
    endif 
    
    ierr = minval(tlev) .lt. 180 ;
    if(ierr) then
        print *,'Temperature is very low -- are you sure RRTMG can handle that?', minval(tlev)
        errcnt = errcnt + 1
    endif 

    ierr = maxval(tlev) .gt. 400 ;
    if(ierr) then 
        print *,'Temperature is very high -- are you sure RRTMG can handle that?', maxval(tlev)
        errcnt = errcnt + 1
    endif 
=======
    integer(iintegers) :: ierr, errcnt
    logical :: lerr

    errcnt = 0
    lerr = maxval(plev) .gt. 1050
    if(lerr) then
      print *,'Pressure above 1050 hPa -- are you sure this is earth?', maxval(plev)
      errcnt = errcnt+1
    endif

    lerr = minval(plev) .lt. zero
    if(lerr) then
      print *,'Pressure negative -- are you sure this is physically correct?', minval(plev)
      errcnt = errcnt+1
    endif

    lerr = minval(tlev) .lt. 180
    if(lerr) then
      print *,'Temperature is very low -- are you sure RRTMG can handle that?', minval(tlev)
      errcnt = errcnt+1
    endif

    lerr = maxval(tlev) .gt. 400
    if(lerr) then
      print *,'Temperature is very high -- are you sure RRTMG can handle that?', maxval(tlev)
      errcnt = errcnt+1
    endif
>>>>>>> upstream/master

    if(errcnt.gt.0) then
      print *,'Found wonky input to tenstream_rrtm_lw -- please check! -- will abort now.'
      call CHKERR(errcnt)
    endif

  end subroutine

  subroutine init_tenstream_rrtm_lw(comm, dx, dy, dz, &
                  phi0, theta0, atm_filename,         &
                  xm, ym, zm, nxproc, nyproc)

    integer(mpiint), intent(in) :: comm

    real(ireals), intent(in) :: dx, dy, phi0, theta0, dz(:,:,:)
    character(default_str_len), intent(in) :: atm_filename
    integer(iintegers),intent(in) :: xm, ym, zm

    integer(iintegers),intent(in), optional :: nxproc(:), nyproc(:) ! array containing xm and ym for all nodes :: dim[x-ranks, y-ranks]

    if(present(nxproc) .neqv. present(nxproc)) then
      print *,'Wrong call to init_tenstream_rrtm_lw --    &
            & in order to work, we need both arrays for &
            & the domain decomposition, call with nxproc AND nxproc'
      stop 'init_tenstream_rrtm_lw -- missing arguments nxproc,nyproc'
    endif
    if(present(nxproc) .and. present(nyproc)) then
      call init_tenstream(comm, zm, xm, ym, dx,dy,phi0, theta0, nxproc=nxproc, nyproc=nyproc, dz3d=dz)
    else ! we let petsc decide where to put stuff
      call init_tenstream(comm, zm, xm, ym, dx, dy, phi0, theta0, dz3d=dz)
    endif

  end subroutine

  subroutine tenstream_rrtm_lw(comm, dx, dy, phi0, theta0, albedo, atm_filename, &
                               edn,eup,abso,                                     &
                               d_plev, d_tlev, d_tlay, d_h2ovmr, d_o3vmr,        &
                               d_co2vmr, d_ch4vmr, d_n2ovmr,  d_o2vmr,           &
                               d_lwc, d_reliq, nxproc, nyproc, icollapse)
    integer(mpiint), intent(in) :: comm

    real(ireals), intent(in) :: dx, dy, phi0, theta0, albedo
    character(default_str_len), intent(in) :: atm_filename

    real(ireals),intent(in) :: d_plev(:,:,:), d_tlev (:,:,:) ! dim(nlay_dynamics+1, nxp, nyp)

    real(ireals),intent(in),optional :: d_tlay   (:,:,:) ! all have
    real(ireals),intent(in),optional :: d_h2ovmr (:,:,:) ! dim(nlay_dynamics, nxp, nyp)
    real(ireals),intent(in),optional :: d_o3vmr  (:,:,:) !
    real(ireals),intent(in),optional :: d_co2vmr (:,:,:) !
    real(ireals),intent(in),optional :: d_ch4vmr (:,:,:) !
    real(ireals),intent(in),optional :: d_n2ovmr (:,:,:) !
    real(ireals),intent(in),optional :: d_o2vmr  (:,:,:) !
    real(ireals),intent(in),optional :: d_lwc    (:,:,:) !
    real(ireals),intent(in),optional :: d_reliq  (:,:,:) !

    integer(iintegers),intent(in),optional :: nxproc(:), nyproc(:), icollapse

    real(rb),allocatable :: col_plev   (:,:)
    real(rb),allocatable :: col_tlev   (:,:)
    real(rb),allocatable :: col_tlay   (:,:)
    real(rb),allocatable :: col_h2ovmr (:,:)
    real(rb),allocatable :: col_o3vmr  (:,:)
    real(rb),allocatable :: col_co2vmr (:,:)
    real(rb),allocatable :: col_ch4vmr (:,:)
    real(rb),allocatable :: col_n2ovmr (:,:)
    real(rb),allocatable :: col_o2vmr  (:,:)
    real(rb),allocatable :: col_lwc    (:,:)
    real(rb),allocatable :: col_lwp    (:,:)
    real(rb),allocatable :: col_reliq  (:,:)

    integer(iintegers) :: i, j, k, icol, ib
    integer(iintegers) :: is,ie, js,je, ks,ke,ke1

    real(ireals),allocatable :: dz(:,:,:)
    real(ireals),allocatable, dimension(:,:,:)   :: col_tau, col_Bfrac                    ! [ncol, nlyr, ngptlw]
    real(ireals),allocatable, dimension(:,:,:)   :: ksca,g                                ! [nlyr, local_nx, local_ny, ngptlw]
    real(ireals),allocatable, dimension(:,:,:,:) :: kabs,Bfrac                            ! [nlyr, local_nx, local_ny, ngptlw]
    real(ireals),allocatable, dimension(:,:,:,:) :: Blev                                  ! [nlyr+1, local_nx, local_ny, nbndlw]
    real(ireals),allocatable, dimension(:,:,:)   :: spec_edir, spec_edn,spec_eup,spec_abso! [nlyr(+1), local_nx, local_ny ]

    real(ireals),allocatable, dimension(:,:,:), intent(out) :: edn,eup,abso               ! [nlyr(+1), local_nx, local_ny ]

    ! character(default_str_len) :: output_path(2) ! [ filename, varname ]

    call load_atmfile(comm, atm_filename, bg_atm)

    call sanitize_input(bg_atm%plev, bg_atm%tlev, bg_atm%tlay)
    do j=lbound(d_plev,3),ubound(d_plev,3)
      do i=lbound(d_plev,2),ubound(d_plev,2)
        if(present(d_tlay)) then
          call sanitize_input(d_plev(:,i,j), d_tlev(:,i,j), d_tlay(:,i,j))
        else
          call sanitize_input(d_plev(:,i,j), d_tlev(:,i,j))
        endif
      enddo
    enddo
    !output_path(1) = 'output.nc'
    !if(myid.eq.0) then
    !  output_path(2) = 'dz3d' ; call ncwrite(output_path, dz, i)
    !  output_path(2) = 'hhl'  ; call ncwrite(output_path, hhl, i)
    !  output_path(2) = 'hsrfc'; call ncwrite(output_path, hhl(ubound(hhl,1),:,:), i)
    !endif

    call merge_dyn_rad_grid(comm, bg_atm,    &
      d_plev, d_tlev, d_tlay, d_h2ovmr,      &
      d_o3vmr, d_co2vmr, d_ch4vmr, d_n2ovmr, &
      d_o2vmr, d_lwc, d_reliq,               &
      col_plev, col_tlev, col_tlay,          &
      col_h2ovmr, col_o3vmr , col_co2vmr,    &
      col_ch4vmr, col_n2ovmr, col_o2vmr ,    &
      col_lwc, col_reliq)


    is = lbound(d_plev,2)  ; ie  = ubound(d_plev,2)
    js = lbound(d_plev,3)  ; je  = ubound(d_plev,3)
    ks = lbound(col_plev,2); ke1 = ubound(col_plev,2); ke=ke1-1

    ! Compute dz on merged grid
    allocate(dz(ke, ie, je))
    do j=js,je
      do i=is,ie
        icol =  i+(j-1)*ie
        dz(:,i,j) = hydrostat_dz_rb(abs(col_plev(icol,1:ke) - col_plev(icol,2:ke1)), &
                                 (col_plev(icol,1:ke) + col_plev(icol,2:ke1))/2,  &
                                 col_tlay(icol,:))
      enddo
    enddo

    if(ldebug .and. myid.eq.0) then
      print *,ke1,'plev', col_plev(1, ke1), 'Tlev', col_tlev(1, ke1)
      do k=ke,1,-1
        print *,k,'dz',dz(k,is,js), 'plev', col_plev(1, k), 'Tlev', col_tlev(1, k), 'Tlay', col_tlay(1, k), 'H2O', col_h2ovmr(1, k), &
          'CO2', col_co2vmr(1, k),'O3', col_o3vmr(1, k),'N2O', col_n2ovmr(1,k),'O2', col_o2vmr(1, k)
      enddo
    endif

    ! RRTMG use liq. water path, not mixing ratio
    allocate(col_lwp(ie*je, ke))
    do j=js,je
      do i=is,ie
        icol =  i+(j-1)*ie
        col_lwp(icol,:) = col_lwc(icol,:) * dz(:,i,j)
      enddo
    enddo
    deallocate(col_lwc)

    ! Compute optical properties with RRTMG
    allocate(col_tau  (ie*je, ke, ngptlw))
    allocate(col_Bfrac(ie*je, ke, ngptlw))

    call optprop_rrtm_lw(ie*je, ke, albedo, &
      col_plev, col_tlev, col_tlay,         &
      col_h2ovmr, col_o3vmr , col_co2vmr,   &
      col_ch4vmr, col_n2ovmr, col_o2vmr ,   &
      col_lwp, col_reliq,                   &
      col_tau, col_Bfrac)


    allocate(kabs (ke , is:ie, js:je, ngptlw))
    allocate(ksca (ke , is:ie, js:je))
    allocate(g    (ke , is:ie, js:je))
    allocate(Bfrac(ke1, is:ie, js:je, ngptlw))
    allocate(Blev (ke1, is:ie, js:je, nbndlw))


    ! rrtmg_lw does not support thermal scattering... set to zero
    ksca = zero
    g    = zero

    do j=js,je
      do i=is,ie
        icol =  i+(j-1)*ie

        ! reverse dz
        dz(:,i,j) = rev1d(dz(:,i,j))

        ! copy from number columns of rrtm interface back onto regular grid
        kabs(:,i,j,:) = max(zero, col_tau(icol,:,:))

        ! divide by thickness to convert from tau to coefficients per meter
        do ib=1, ngptlw
          kabs(:,i,j,ib) = rev1d(kabs(:,i,j,ib)) / dz(:,i,j)

          Bfrac(2:ke1,i,j,ib) = col_Bfrac(icol,:,ib)
          Bfrac(1,i,j,ib) = Bfrac(2,i,j,ib) ! surface weights are the same as lowest layer

          Bfrac(:,i,j,ib) = rev1d(Bfrac(:,i,j,ib))
        enddo

        ! Compute source term(planck function)
        do ib=1,nbndlw
          do k=1,ke
            Blev(k+1,i,j,ib) = plkint(real(wavenum1(ib)), real(wavenum2(ib)), real(col_tlay(icol,k)))
          enddo
          Blev(1,i,j,ib) = plkint(real(wavenum1(ib)), real(wavenum2(ib)), real(col_tlev(icol,1)))

          ! col_tlxx starts at surface but tenstream takes planck function
          ! starting at top --> reverse
          Blev(:,i,j,ib) = rev1d(Blev(:,i,j,ib))
        enddo

      enddo
    enddo

    ! Free up some intermediate memory
    deallocate(col_plev  )
    deallocate(col_tlev  )
    deallocate(col_tlay  )
    deallocate(col_h2ovmr)
    deallocate(col_o3vmr )
    deallocate(col_co2vmr)
    deallocate(col_ch4vmr)
    deallocate(col_n2ovmr)
    deallocate(col_o2vmr )
    deallocate(col_lwp   )
    deallocate(col_reliq )
    deallocate(col_tau   )
    deallocate(col_Bfrac )


    if(.not.linit_tenstr) then
      call init_tenstream_rrtm_lw(comm, dx, dy, dz, phi0, theta0, atm_filename, &
        ie,je,ke, nxproc, nyproc)
      linit_tenstr=.True.
    endif

    ! Allocate space for results -- for integrated values and for temporary spectral integration...
    allocate(edn (C_one1%zm, C_one1%xm, C_one1%ym), source=zero)
    allocate(eup (C_one1%zm, C_one1%xm, C_one1%ym), source=zero)
    allocate(abso(C_one%zm , C_one%xm , C_one%ym ), source=zero)

    allocate(spec_edn (C_one1%zm, C_one1%xm, C_one1%ym))
    allocate(spec_eup (C_one1%zm, C_one1%xm, C_one1%ym))
    allocate(spec_abso(C_one%zm , C_one%xm , C_one%ym ))

    ! Loop over spectral intervals and call solver
    do ib=1,ngptlw
      call set_optical_properties(albedo, kabs(:,:,:,ib), ksca(:,:,:), g(:,:,:), Blev(:,:,:,ngb(ib))*Bfrac(:,:,:,ib))
      call solve_tenstream(zero)
      call tenstream_get_result(spec_edir, spec_edn, spec_eup, spec_abso)

      edn  = edn  + spec_edn
      eup  = eup  + spec_eup
      abso = abso + spec_abso
    enddo

  end subroutine
  subroutine destroy_tenstream_rrtm_lw()
    ! Tidy up the solver
    call destroy_tenstream(lfinalizepetsc=.True.)
    linit_tenstr = .False.
  end subroutine

  subroutine optprop_rrtm_lw(ncol_in, nlay_in, albedo, plev, tlev, tlay, h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, lwp, reliq, tau, Bfrac)
    ! RRTM needs the arrays to start at the surface

    integer(iintegers),intent(in)          :: ncol_in, nlay_in
    real(ireals), intent(in) :: albedo

    real(rb),dimension(ncol_in,nlay_in+1) :: plev, tlev
    real(rb),dimension(ncol_in,nlay_in)   :: tlay, h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr
    real(rb),dimension(ncol_in,nlay_in)   :: lwp, reliq

    real(ireals), dimension(:,:,:), intent(out) :: tau, Bfrac ! [ncol, nlay, ngptlw]

    real(rb),dimension(ncol_in,nlay_in) :: play, cldfr, cicewp, reice

    real(rb),dimension(nbndlw, ncol_in, nlay_in) :: taucld
    real(rb),dimension(ncol_in, nlay_in, nbndlw ) :: tauaer
    real(rb),dimension(ncol_in, nbndlw ) :: emis

    real(rb),dimension(ncol_in, nlay_in)   :: cfc11vmr,cfc12vmr,cfc22vmr,ccl4vmr

    real(rb),dimension(ncol_in) :: tsfc

    real(rb),dimension(ncol_in,nlay_in+1) :: lwuflx,lwdflx,lwuflxc,lwdflxc
    real(rb),dimension(ncol_in,nlay_in  ) :: lwhr,lwhrc

    integer(im) :: k,icol
    integer(im) :: ncol, nlay

    integer(im),parameter :: inflglw=2,iceflglw=3,liqflglw=1
    integer(kind=im) :: icld=2         ! Cloud overlap method
    integer(kind=im) :: iaer=0         ! Aerosol option flag
    integer(kind=im) :: idrv=0         ! Flag for calculation of dFdT

    logical,save :: linit_rrtmg=.False.

    ! copy from TenStream to RRTM precision:
    ncol   = ncol_in
    nlay   = nlay_in

    ! Take average pressure and temperature as mean values for voxels --
    ! should probably use log interpolation for pressure...
    do icol=1,ncol
      play(icol,:) = .5_rb*(plev(icol,1:nlay)+plev(icol,2:nlay+1))

      tsfc(icol)   = tlev(icol,1)
    enddo

    taucld   = 0; cicewp   = 0; reice    = 0;
    tauaer   = 0; lwdflxc  = 0; lwuflxc  = 0;
    cfc11vmr = 0; cfc12vmr = 0; cfc22vmr = 0; ccl4vmr = 0;

    emis = one - albedo

    where ( lwp.gt.0 )
      cldfr = 1
    elsewhere
      cldfr = 0
    endwhere

    if(.not.linit_rrtmg) then
      call rrtmg_lw_ini(1006._rb)
      linit_rrtmg = .True.

      if(ldebug .and. myid.eq.0) then
        do k=nlay,1,-1
          print *,'rrtm_optprop_lw',k,'tlev',tlev(1,k),'tlay',tlay(1,k),'plev',plev(1,k),'play',play(1,k),'lwp',lwp(1,k), &
            'reliq',reliq(1,k), 'h2o',h2ovmr(1,k), 'o3' , o3vmr(1,k), 'co2', co2vmr(1,k), 'ch4', ch4vmr(1,k),    &
            'n2o', n2ovmr(1,k), 'o2' , o2vmr(1,k)
        enddo
      endif
    endif

    call rrtmg_lw &
      (ncol    ,nlay    ,icld    ,idrv   , &
      play    ,plev    ,tlay    ,tlev    ,tsfc    , &
      h2ovmr  ,o3vmr   ,co2vmr  ,ch4vmr  ,n2ovmr  ,o2vmr, &
      cfc11vmr,cfc12vmr,cfc22vmr,ccl4vmr ,emis    , &
      inflglw, iceflglw, liqflglw, cldfr, &
      taucld , cicewp  , lwp  ,reice   ,reliq      , &
      tauaer , &
      lwuflx , lwdflx  ,lwhr    ,lwuflxc ,lwdflxc ,lwhrc, &
      tau, Bfrac)
  end subroutine

  subroutine hydrostat_lev(plev,tlay, hsrfc, hhl, dz)
    ! Integrate vertical height profile hydrostatically. arrays start at bottom(surface)
    real(ireals),intent(in) :: plev(:),tlay(:)
    real(ireals),intent(in) :: hsrfc
    real(ireals),intent(out) :: hhl(size(plev))
    real(ireals),intent(out) :: dz(size(tlay))
    integer(im) :: k
    hhl(1) = hsrfc
    do k=1,size(tlay)
      dz(k) = hydrostat_dz_ireal(abs(plev(k+1)-plev(k)), (plev(k+1)+plev(k))/2, tlay(k))
      hhl(k+1) = hhl(k) + dz(k)
    enddo
    if(any(dz.le.zero)) then
      print *,'plev',plev
      print *,'tlay',tlay
      print *,'dz',dz
      print *,'hhl',hhl
      stop 'error in dz'
    endif
  end subroutine

  pure elemental function hydrostat_dz_ireal(dp, p, T)
    real(ireals), intent(in) :: dp, p, T
    real(ireals) :: hydrostat_dz_ireal, rho
    rho = p / 287.058_ireals / T
    hydrostat_dz_ireal = dp / rho / 9.8065_ireals
  end function
  pure elemental function hydrostat_dz_rb(dp, p, T)
    real(rb), intent(in) :: dp, p, T
    real(rb) :: hydrostat_dz_rb, rho
    rho = p / 287.058_rb / T
    hydrostat_dz_rb = dp / rho / 9.8065_rb
  end function

  function rev2d(inp) ! reverse second dimension
    real(ireals),intent(in) :: inp(:,:)
    real(rb) :: rev2d(size(inp,1),size(inp,2))
    rev2d = inp(:,ubound(inp,2):lbound(inp,2):-1)
  end function

  function rev1d(inp) ! reverse array
    real(ireals),intent(in) :: inp(:)
    real(ireals) :: rev1d(size(inp,1))
    rev1d = inp(ubound(inp,1):lbound(inp,1):-1)
  end function

  subroutine load_atmfile(comm, atm_filename, atm)
    integer(mpiint), intent(in) :: comm
    character(default_str_len), intent(in) :: atm_filename
    type(t_atm),allocatable,intent(inout) :: atm

    integer(mpiint) :: myid
    integer(iintegers) :: k, nlev
    real(ireals),allocatable :: prof(:,:) ! # z(km)  p(mb)  T(K) air(cm-3) o3(cm-3) o2(cm-3)  h2o(cm-3) co2(cm-3) no2(cm-3)

    if(allocated(atm)) return

    call mpi_comm_rank(comm, myid, mpierr)
    allocate(atm)

    if(myid.eq.0) then
      call read_ascii_file_2d(atm_filename, prof, 9, 2, ierr); call CHKERR(ierr)

      nlev = ubound(prof,1)

      allocate(atm%plev   (nlev))
      allocate(atm%zt     (nlev))
      allocate(atm%tlev   (nlev))
      allocate(atm%h2o_lev(nlev))
      allocate(atm%o3_lev (nlev))
      allocate(atm%co2_lev(nlev))
      allocate(atm%ch4_lev(nlev))
      allocate(atm%n2o_lev(nlev))
      allocate(atm%o2_lev (nlev))

      atm%zt   = prof(:,1)*1e3
      atm%plev = prof(:,2)
      atm%tlev = prof(:,3)
      atm%h2o_lev = prof(:,7) / prof(:,4)
      atm%o3_lev  = prof(:,5) / prof(:,4)
      atm%co2_lev = prof(:,8) / prof(:,4)
      atm%ch4_lev = atm%co2_lev / 1e2
      atm%n2o_lev = prof(:,9) / prof(:,4)
      atm%o2_lev  = prof(:,6) / prof(:,4)

      if(ldebug .and. myid.eq.0) then
        do k=1, nlev
          print *,k,'zt', atm%zt(k), 'plev', atm%plev(k), 'T', atm%tlev(k), 'CO2', atm%co2_lev(k), 'H2O', atm%h2o_lev(k), 'O3', atm%o3_lev(k),'N2O' , atm%n2o_lev(k), 'O2', atm%o2_lev(k)
        enddo

      endif
    endif
    call imp_bcast(comm, atm%plev   , 0_mpiint)
    call imp_bcast(comm, atm%zt     , 0_mpiint)
    call imp_bcast(comm, atm%tlev   , 0_mpiint)
    call imp_bcast(comm, atm%h2o_lev, 0_mpiint)
    call imp_bcast(comm, atm%o3_lev , 0_mpiint)
    call imp_bcast(comm, atm%co2_lev, 0_mpiint)
    call imp_bcast(comm, atm%ch4_lev, 0_mpiint)
    call imp_bcast(comm, atm%n2o_lev, 0_mpiint)
    call imp_bcast(comm, atm%o2_lev , 0_mpiint)

    nlev = size(atm%plev)

    allocate(atm%play   (nlev-1))
    allocate(atm%zm     (nlev-1))
    allocate(atm%dz     (nlev-1))
    allocate(atm%tlay   (nlev-1))
    allocate(atm%h2o_lay(nlev-1))
    allocate(atm%o3_lay (nlev-1))
    allocate(atm%co2_lay(nlev-1))
    allocate(atm%ch4_lay(nlev-1))
    allocate(atm%n2o_lay(nlev-1))
    allocate(atm%o2_lay (nlev-1))

    atm%play    = meanvec(atm%plev   )
    atm%zm      = meanvec(atm%zt     )
    atm%dz      = atm%zt(1:nlev-1) - atm%zt(2:nlev)
    atm%tlay    = meanvec(atm%tlev   )
    atm%h2o_lay = meanvec(atm%h2o_lev)
    atm%o3_lay  = meanvec(atm%o3_lev )
    atm%co2_lay = meanvec(atm%co2_lev)
    atm%ch4_lay = meanvec(atm%ch4_lev)
    atm%n2o_lay = meanvec(atm%n2o_lev)
    atm%o2_lay  = meanvec(atm%o2_lev )
  end subroutine
  subroutine merge_dyn_rad_grid(comm, atm,   &
      d_plev, d_tlev, d_tlay, d_h2ovmr,      &
      d_o3vmr, d_co2vmr, d_ch4vmr, d_n2ovmr, &
      d_o2vmr, d_lwc, d_reliq,               &
      col_plev, col_tlev, col_tlay,          &
      col_h2ovmr, col_o3vmr , col_co2vmr,    &
      col_ch4vmr, col_n2ovmr, col_o2vmr ,    &
      col_lwc, col_reliq)

    integer(mpiint), intent(in) :: comm
    type(t_atm),intent(in) :: atm ! 1D background profile info

    real(ireals),intent(in) :: d_plev (:,:,:), d_tlev(:,:,:) ! dim(nlay_dynamics+1, nxp, nyp)

    real(ireals),intent(in),optional :: d_tlay   (:,:,:) ! all have
    real(ireals),intent(in),optional :: d_h2ovmr (:,:,:) ! dim(nlay_dynamics, nxp, nyp)
    real(ireals),intent(in),optional :: d_o3vmr  (:,:,:) !
    real(ireals),intent(in),optional :: d_co2vmr (:,:,:) !
    real(ireals),intent(in),optional :: d_ch4vmr (:,:,:) !
    real(ireals),intent(in),optional :: d_n2ovmr (:,:,:) !
    real(ireals),intent(in),optional :: d_o2vmr  (:,:,:) !
    real(ireals),intent(in),optional :: d_lwc    (:,:,:) !
    real(ireals),intent(in),optional :: d_reliq  (:,:,:) !

    real(rb),intent(out),allocatable :: col_plev   (:,:)
    real(rb),intent(out),allocatable :: col_tlev   (:,:)
    real(rb),intent(out),allocatable :: col_tlay   (:,:)
    real(rb),intent(out),allocatable :: col_h2ovmr (:,:)
    real(rb),intent(out),allocatable :: col_o3vmr  (:,:)
    real(rb),intent(out),allocatable :: col_co2vmr (:,:)
    real(rb),intent(out),allocatable :: col_ch4vmr (:,:)
    real(rb),intent(out),allocatable :: col_n2ovmr (:,:)
    real(rb),intent(out),allocatable :: col_o2vmr  (:,:)
    real(rb),intent(out),allocatable :: col_lwc    (:,:)
    real(rb),intent(out),allocatable :: col_reliq  (:,:)

    integer(iintegers) :: d_ke, d_ke1 ! number of vertical levels of dynamics grid
    integer(iintegers) :: atm_ke      ! number of vertical levels of atmosphere grid

    integer(iintegers) :: ke, ke1 ! number of vertical levels of merged grid
    integer(iintegers) :: is,ie, js,je, icol
    integer(iintegers) :: i, j

    real(ireals),allocatable :: d_hhl(:,:,:), d_dz(:)
    real(ireals) :: global_maxheight, global_minplev

    is = lbound(d_plev,2); ie = ubound(d_plev,2)
    js = lbound(d_plev,3); je = ubound(d_plev,3)

    d_ke1 = ubound(d_plev,1); d_ke = d_ke1-1

    ! find out how many layers we have to put on top of the dynamics grid

    ! First get top height of dynamics grid
    allocate(d_hhl(d_ke1, ie, je))
    allocate(d_dz(d_ke))

    do j=js,je
      do i=is,ie
        if(present(d_tlay)) then
          call hydrostat_lev(d_plev(:,i,j),d_tlay(:,i,j), zero, d_hhl(:, i,j), d_dz)
        else
          call hydrostat_lev(d_plev(:,i,j),(d_tlev(1:d_ke,i,j)+d_tlev(2:d_ke1,i,j))/2, zero, d_hhl(:, i,j), d_dz)
        endif
      enddo
    enddo

    ! index of lowermost layer in atm: search for level where height is bigger and
    ! pressure is lower
    call imp_allreduce_max(comm, maxval(d_hhl), global_maxheight)
    call imp_allreduce_min(comm, minval(d_plev), global_minplev)

    i = floor(search_sorted_bisection(atm%zt, global_maxheight))
    j = floor(search_sorted_bisection(atm%plev, global_minplev))
    atm_ke = min(i,j)
    ke  = atm_ke + d_ke
    ke1 = atm_ke + d_ke1

    ! then from there on couple background atm data on top of that

    allocate(col_plev   (ie*je, ke1))

    allocate(col_tlev   (ie*je, ke1))
    allocate(col_tlay   (ie*je, ke ))
    allocate(col_h2ovmr (ie*je, ke ))
    allocate(col_o3vmr  (ie*je, ke ))
    allocate(col_co2vmr (ie*je, ke ))
    allocate(col_ch4vmr (ie*je, ke ))
    allocate(col_n2ovmr (ie*je, ke ))
    allocate(col_o2vmr  (ie*je, ke ))
    allocate(col_lwc    (ie*je, ke ))
    allocate(col_reliq  (ie*je, ke ))
    do j=js,je
      do i=is,ie
        icol = i+(j-1)*ie

        ! First merge pressure levels .. pressure is always given..
        col_plev(icol, ke1-atm_ke+1:ke1) = rev1d(atm%plev(1:atm_ke))
        col_plev(icol, 1:d_ke1) = d_plev(:,i,j)
        if(col_plev(icol, ke1-atm_ke+1) .ge. col_plev(icol,d_ke1)) then
          print *,'background profile pressure is .ge. than uppermost pressure &
            & level of dynamics grid -- this suggests the dynamics grid is way &
            & off hydrostatic balance... please check', col_plev(icol,:)
          stop 'error in rrtm_lw merging grids'
        endif

        ! And also Tlev has to be present always
        col_tlev(icol, ke1-atm_ke+1:ke1) = rev1d(atm%tlev(1:atm_ke))
        col_tlev(icol, 1:d_ke1) = d_tlev(:,i,j)

        if(present(d_tlay)) then
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%tlay, atm%tlev, col_tlay(icol,:), d_tlay(:,i,j))
        else
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%tlay, atm%tlev, col_tlay(icol,:), (d_tlev(1:d_ke,i,j)+d_tlev(2:d_ke1,i,j))/2)
        endif

        if(present(d_lwc)) then
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, zero*atm%tlay, zero*atm%tlev, col_lwc(icol,:), d_lwc(:,i,j))
        else
          col_lwc(icol,:) = 0
        endif
        if(present(d_reliq)) then
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, zero*atm%tlay, zero*atm%tlev, col_reliq(icol,:), d_reliq(:,i,j))
        else
          col_reliq = zero
        endif

        if(present(d_h2ovmr)) then
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%h2o_lay, atm%h2o_lev, col_h2ovmr(icol,:), d_h2ovmr(:,i,j))
        else
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%h2o_lay, atm%h2o_lev, col_h2ovmr(icol,:))
        endif
        if(present(d_o3vmr)) then
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%o3_lay, atm%o3_lev, col_o3vmr(icol,:), d_o3vmr(:,i,j))
        else
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%o3_lay, atm%o3_lev, col_o3vmr(icol,:))
        endif

        if(present(d_co2vmr)) then
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%co2_lay, atm%co2_lev, col_co2vmr(icol,:), d_co2vmr(:,i,j))
        else
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%co2_lay, atm%co2_lev, col_co2vmr(icol,:))
        endif
        if(present(d_ch4vmr)) then
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%ch4_lay, atm%ch4_lev, col_ch4vmr(icol,:), d_ch4vmr(:,i,j))
        else
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%ch4_lay, atm%ch4_lev, col_ch4vmr(icol,:))
        endif
        if(present(d_n2ovmr)) then
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%n2o_lay, atm%n2o_lev, col_n2ovmr(icol,:), d_n2ovmr(:,i,j))
        else
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%n2o_lay, atm%n2o_lev, col_n2ovmr(icol,:))
        endif
        if(present(d_o2vmr)) then
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%o2_lay, atm%o2_lev, col_o2vmr(icol,:), d_o2vmr(:,i,j))
        else
          call merge_grid_var(atm%zt, d_hhl(:,i,j), atm_ke, atm%o2_lay, atm%o2_lev, col_o2vmr(icol,:))
        endif
      enddo
    enddo

  end subroutine


  ! merge the dynamics grid and the background profile together at lvl atm_ke
  ! NOTE! Only use with variables on layer
  subroutine merge_grid_var(a_hhl, d_hhl, atm_ke, a_lay, a_lev, col_var, d_var)
    integer(iintegers),intent(in) :: atm_ke
    real(ireals),intent(in) :: a_hhl(:), d_hhl(:), a_lay(:), a_lev(:) ! a_arr is from atm%, d_arr corresponds to dynamics grids
    real(rb),intent(out) :: col_var(:)
    real(ireals),intent(in),optional :: d_var(:)
    integer(iintegers) :: k, kt ! kt is reverse index
    real(ireals) :: h

    ! Top of atmosphere layers are always given by background profile
    do k=1,atm_ke
      kt = size(col_var)-k+1
      col_var(kt) = a_lay(k)
    enddo

    if(present(d_var)) then ! dynamics grid variable is provided, use that
      do k=1,size(d_var)
        col_var(k) = d_var(k)
      enddo
    else ! we may still use atmospheric grid file instead...
      do k=1,size(col_var)-atm_ke
        h = (d_hhl(k+1) + d_hhl(k)) / 2
        col_var(k) = interp_1d(search_sorted_bisection(a_hhl, h), a_lev)
      enddo
    endif

  end subroutine
end module
