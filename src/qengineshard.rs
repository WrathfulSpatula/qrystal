use std::collections::HashMap;
use std::rc::Rc;

type Complex = num::Complex<f64>;

const ONE_CMPLX: Complex = Complex { re: 1.0, im: 0.0 };
const ZERO_CMPLX: Complex = Complex { re: 0.0, im: 0.0 };
const I_CMPLX: Complex = Complex { re: 0.0, im: 1.0 };

struct PhaseShard {
    cmplxDiff: Complex,
    cmplxSame: Complex,
    isInvert: bool,
}

impl PhaseShard {
    fn new() -> Self {
        PhaseShard {
            cmplxDiff: ONE_CMPLX,
            cmplxSame: ONE_CMPLX,
            isInvert: false,
        }
    }
}

struct QEngineShard {
    unit: Option<Rc<QInterface>>,
    mapped: usize,
    isProbDirty: bool,
    isPhaseDirty: bool,
    amp0: Complex,
    amp1: Complex,
    pauliBasis: Pauli,
    controlsShards: HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
    antiControlsShards: HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
    targetOfShards: HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
    antiTargetOfShards: HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
    found: bool,
}

impl QEngineShard {
    fn new() -> Self {
        QEngineShard {
            unit: None,
            mapped: 0,
            isProbDirty: false,
            isPhaseDirty: false,
            amp0: ONE_CMPLX,
            amp1: ZERO_CMPLX,
            pauliBasis: PauliZ,
            controlsShards: HashMap::new(),
            antiControlsShards: HashMap::new(),
            targetOfShards: HashMap::new(),
            antiTargetOfShards: HashMap::new(),
            found: false,
        }
    }

    fn new_with_set(set: bool, rand_phase: Complex) -> Self {
        let mut shard = QEngineShard::new();
        shard.amp0 = if set { ZERO_CMPLX } else { rand_phase };
        shard.amp1 = if set { rand_phase } else { ZERO_CMPLX };
        shard
    }

    fn new_with_u_mapping(u: Rc<QInterface>, mapping: usize) -> Self {
        QEngineShard {
            unit: Some(u),
            mapped: mapping,
            isProbDirty: true,
            isPhaseDirty: true,
            amp0: ONE_CMPLX,
            amp1: ZERO_CMPLX,
            pauliBasis: PauliZ,
            controlsShards: HashMap::new(),
            antiControlsShards: HashMap::new(),
            targetOfShards: HashMap::new(),
            antiTargetOfShards: HashMap::new(),
            found: false,
        }
    }

    fn make_dirty(&mut self) {
        self.isProbDirty = true;
        self.isPhaseDirty = true;
    }

    fn clamp_amps(&mut self) -> bool {
        // TODO: Implement this function
        false
    }

    fn remove_buffer(
        &mut self,
        p: Rc<QEngineShard>,
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_map_get: fn(&QEngineShard) -> &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
    ) {
        if let Some(phase_shard) = local_map.remove(&p) {
            let remote_map = remote_map_get(&*phase_shard);
            remote_map.remove(self);
        }
    }

    fn remove_control(&mut self, p: Rc<QEngineShard>) {
        self.remove_buffer(p, &mut self.targetOfShards, &QEngineShard::get_controls_shards);
    }

    fn remove_target(&mut self, p: Rc<QEngineShard>) {
        self.remove_buffer(p, &mut self.controlsShards, &QEngineShard::get_target_of_shards);
    }

    fn remove_anti_control(&mut self, p: Rc<QEngineShard>) {
        self.remove_buffer(p, &mut self.antiTargetOfShards, &QEngineShard::get_anti_controls_shards);
    }

    fn remove_anti_target(&mut self, p: Rc<QEngineShard>) {
        self.remove_buffer(p, &mut self.antiControlsShards, &QEngineShard::get_anti_target_of_shards);
    }

    fn dump_buffer(
        &mut self,
        optimize_fn: fn(&mut QEngineShard),
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_fn: fn(&mut QEngineShard, Rc<QEngineShard>),
    ) {
        (optimize_fn)(self);
        let mut phase_shard = local_map.iter().next();
        while let Some((partner, _)) = phase_shard {
            (remote_fn)(self, partner.clone());
            phase_shard = local_map.iter().next();
        }
    }

    fn dump_same_phase_buffer(
        &mut self,
        optimize_fn: fn(&mut QEngineShard),
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_fn: fn(&mut QEngineShard, Rc<QEngineShard>),
    ) {
        (optimize_fn)(self);
        let mut phase_shard = local_map.iter().next();
        let mut lcv = 0;
        while let Some((partner, buffer)) = phase_shard {
            if !buffer.is_invert && IS_SAME(buffer.cmplxDiff, buffer.cmplxSame) {
                (remote_fn)(self, partner.clone());
            } else {
                lcv += 1;
            }
            phase_shard = local_map.iter().nth(lcv);
        }
    }

    fn dump_control_of(&mut self) {
        self.dump_buffer(
            &QEngineShard::optimize_targets,
            &mut self.controlsShards,
            &QEngineShard::remove_target,
        );
    }

    fn dump_anti_control_of(&mut self) {
        self.dump_buffer(
            &QEngineShard::optimize_anti_targets,
            &mut self.antiControlsShards,
            &QEngineShard::remove_anti_target,
        );
    }

    fn dump_same_phase_control_of(&mut self) {
        self.dump_same_phase_buffer(
            &QEngineShard::optimize_targets,
            &mut self.controlsShards,
            &QEngineShard::remove_target,
        );
    }

    fn dump_same_phase_anti_control_of(&mut self) {
        self.dump_same_phase_buffer(
            &QEngineShard::optimize_anti_targets,
            &mut self.antiControlsShards,
            &QEngineShard::remove_anti_target,
        );
    }

    fn add_buffer(
        &mut self,
        p: Rc<QEngineShard>,
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_fn: fn(&mut QEngineShard, Rc<QEngineShard>),
    ) {
        if !local_map.contains_key(&p) {
            let ps = Rc::new(PhaseShard::new());
            local_map.insert(p.clone(), ps.clone());
            let remote_map = remote_fn(&*ps);
            remote_map.insert(self, ps);
        }
    }

    fn make_phase_controlled_by(&mut self, p: Rc<QEngineShard>) {
        self.add_buffer(p, &mut self.targetOfShards, &QEngineShard::get_controls_shards);
    }

    fn make_phase_control_of(&mut self, p: Rc<QEngineShard>) {
        self.add_buffer(p, &mut self.controlsShards, &QEngineShard::get_target_of_shards);
    }

    fn make_phase_anti_controlled_by(&mut self, p: Rc<QEngineShard>) {
        self.add_buffer(p, &mut self.antiTargetOfShards, &QEngineShard::get_anti_controls_shards);
    }

    fn make_phase_anti_control_of(&mut self, p: Rc<QEngineShard>) {
        self.add_buffer(p, &mut self.antiControlsShards, &QEngineShard::get_anti_target_of_shards);
    }

    fn add_angles(
        &mut self,
        control: Rc<QEngineShard>,
        cmplxDiff: Complex,
        cmplxSame: Complex,
        local_fn: fn(&mut QEngineShard, Rc<QEngineShard>),
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_fn: fn(&mut QEngineShard, Rc<QEngineShard>),
    ) {
        (local_fn)(self, control.clone());
        let target_of_shard = local_map.get_mut(&control).unwrap();
        let ncmplxDiff = target_of_shard.cmplxDiff * cmplxDiff / cmplxDiff.norm();
        let ncmplxSame = target_of_shard.cmplxSame * cmplxSame / cmplxSame.norm();
        if !target_of_shard.isInvert && IS_ARG_0(ncmplxDiff) && IS_ARG_0(ncmplxSame) {
            (remote_fn)(self, control);
            return;
        }
        target_of_shard.cmplxDiff = ncmplxDiff;
        target_of_shard.cmplxSame = ncmplxSame;
    }

    fn add_phase_angles(&mut self, control: Rc<QEngineShard>, topLeft: Complex, bottomRight: Complex) {
        self.add_angles(
            control,
            topLeft,
            bottomRight,
            &QEngineShard::make_phase_controlled_by,
            &mut self.targetOfShards,
            &QEngineShard::remove_control,
        );
    }

    fn add_anti_phase_angles(&mut self, control: Rc<QEngineShard>, bottomRight: Complex, topLeft: Complex) {
        self.add_angles(
            control,
            bottomRight,
            topLeft,
            &QEngineShard::make_phase_anti_controlled_by,
            &mut self.antiTargetOfShards,
            &QEngineShard::remove_anti_control,
        );
    }

    fn add_inversion_angles(&mut self, control: Rc<QEngineShard>, topRight: Complex, bottomLeft: Complex) {
        self.make_phase_controlled_by(control.clone());
        let target_of_shard = self.targetOfShards.get_mut(&control).unwrap();
        target_of_shard.isInvert = !target_of_shard.isInvert;
        std::mem::swap(&mut target_of_shard.cmplxDiff, &mut target_of_shard.cmplxSame);
        self.add_phase_angles(control, topRight, bottomLeft);
    }

    fn add_anti_inversion_angles(&mut self, control: Rc<QEngineShard>, bottomLeft: Complex, topRight: Complex) {
        self.make_phase_anti_controlled_by(control.clone());
        let anti_target_of_shard = self.antiTargetOfShards.get_mut(&control).unwrap();
        anti_target_of_shard.isInvert = !anti_target_of_shard.isInvert;
        std::mem::swap(&mut anti_target_of_shard.cmplxDiff, &mut anti_target_of_shard.cmplxSame);
        self.add_anti_phase_angles(control, bottomLeft, topRight);
    }

    fn optimize_buffer(
        &mut self,
        local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        remote_map_get: fn(&QEngineShard) -> &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        phase_fn: fn(&mut QEngineShard, Rc<QEngineShard>, Complex, Complex),
        make_this_control: bool,
    ) {
        let temp_local_map = local_map.clone();
        for (partner, buffer) in temp_local_map {
            if buffer.isInvert || !IS_ARG_0(buffer.cmplxDiff) {
                continue;
            }
            let remote_map = remote_map_get(&*buffer);
            remote_map.remove(self);
            local_map.remove(&partner);
            if make_this_control {
                (phase_fn)(self, partner, ONE_CMPLX, buffer.cmplxSame);
            } else {
                (phase_fn)(partner, self, ONE_CMPLX, buffer.cmplxSame);
            }
        }
    }

    fn optimize_controls(&mut self) {
        self.optimize_buffer(
            &mut self.controlsShards,
            &QEngineShard::get_target_of_shards,
            &QEngineShard::add_phase_angles,
            false,
        );
    }

    fn optimize_targets(&mut self) {
        self.optimize_buffer(
            &mut self.targetOfShards,
            &QEngineShard::get_controls_shards,
            &QEngineShard::add_phase_angles,
            true,
        );
    }

    fn optimize_anti_controls(&mut self) {
        self.optimize_buffer(
            &mut self.antiControlsShards,
            &QEngineShard::get_anti_target_of_shards,
            &QEngineShard::add_anti_phase_angles,
            false,
        );
    }

    fn optimize_anti_targets(&mut self) {
        self.optimize_buffer(
            &mut self.antiTargetOfShards,
            &QEngineShard::get_anti_controls_shards,
            &QEngineShard::add_anti_phase_angles,
            true,
        );
    }

    fn optimize_both_targets(&mut self) {
        let temp_local_map = self.targetOfShards.clone();
        for (partner, buffer) in temp_local_map {
            if buffer.isInvert {
                continue;
            }
            if IS_ARG_0(buffer.cmplxDiff) {
                partner.get_controls_shards().remove(self);
                self.targetOfShards.remove(&partner);
                partner.add_phase_angles(self, ONE_CMPLX, buffer.cmplxSame);
            } else if IS_ARG_0(buffer.cmplxSame) {
                partner.get_controls_shards().remove(self);
                self.targetOfShards.remove(&partner);
                partner.add_anti_phase_angles(self, buffer.cmplxDiff, ONE_CMPLX);
            }
        }
        let temp_local_map = self.antiTargetOfShards.clone();
        for (partner, buffer) in temp_local_map {
            if buffer.isInvert {
                continue;
            }
            if IS_ARG_0(buffer.cmplxDiff) {
                partner.get_anti_controls_shards().remove(self);
                self.antiTargetOfShards.remove(&partner);
                partner.add_anti_phase_angles(self, ONE_CMPLX, buffer.cmplxSame);
            } else if IS_ARG_0(buffer.cmplxSame) {
                partner.get_anti_controls_shards().remove(self);
                self.antiTargetOfShards.remove(&partner);
                partner.add_phase_angles(self, buffer.cmplxDiff, ONE_CMPLX);
            }
        }
    }

    fn combine_buffers(
        &mut self,
        target_map_get: fn(&QEngineShard) -> &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        control_map_get: fn(&QEngineShard) -> &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>,
        angle_fn: fn(&mut QEngineShard, Rc<QEngineShard>, Complex, Complex),
    ) {
        let temp_controls = control_map_get(self).clone();
        let temp_targets = target_map_get(self).clone();
        for (partner, buffer1) in temp_controls {
            if let Some(buffer2) = temp_targets.get(&partner) {
                if !buffer1.isInvert && IS_ARG_0(buffer1.cmplxDiff) {
                    partner.get_target_of_shards().remove(self);
                    control_map_get(self).remove(&partner);
                    (angle_fn)(partner, self, ONE_CMPLX, buffer1.cmplxSame);
                } else if !buffer2.isInvert && IS_ARG_0(buffer2.cmplxDiff) {
                    partner.get_controls_shards().remove(self);
                    target_map_get(self).remove(&partner);
                    (angle_fn)(self, partner, ONE_CMPLX, buffer2.cmplxSame);
                }
            }
        }
    }

    fn combine_gates(&mut self) {
        self.combine_buffers(
            &QEngineShard::get_target_of_shards,
            &QEngineShard::get_controls_shards,
            &QEngineShard::add_phase_angles,
        );
        self.combine_buffers(
            &QEngineShard::get_anti_target_of_shards,
            &QEngineShard::get_anti_controls_shards,
            &QEngineShard::add_anti_phase_angles,
        );
    }

    fn swap_target_anti(&mut self, control: Rc<QEngineShard>) {
        if let Some(phase_shard) = self.targetOfShards.remove(&control) {
            if let Some(anti_phase_shard) = self.antiTargetOfShards.remove(&control) {
                std::mem::swap(&mut phase_shard.cmplxDiff, &mut anti_phase_shard.cmplxDiff);
                std::mem::swap(&mut phase_shard.cmplxSame, &mut anti_phase_shard.cmplxSame);
                self.targetOfShards.insert(control.clone(), anti_phase_shard);
                self.antiTargetOfShards.insert(control, phase_shard);
            } else {
                std::mem::swap(&mut phase_shard.cmplxDiff, &mut phase_shard.cmplxSame);
                self.antiTargetOfShards.insert(control, phase_shard);
            }
        } else if let Some(anti_phase_shard) = self.antiTargetOfShards.remove(&control) {
            std::mem::swap(&mut anti_phase_shard.cmplxDiff, &mut anti_phase_shard.cmplxSame);
            self.targetOfShards.insert(control, anti_phase_shard);
        }
    }

    fn flip_phase_anti(&mut self) {
        let mut to_swap = std::collections::HashSet::new();
        for (ctrl, _) in &self.controlsShards {
            to_swap.insert(ctrl.clone());
        }
        for (ctrl, _) in &self.antiControlsShards {
            to_swap.insert(ctrl.clone());
        }
        for swap_shard in to_swap {
            swap_shard.swap_target_anti(self);
        }
        std::mem::swap(&mut self.controlsShards, &mut self.antiControlsShards);
        for (_, phase_shard) in &mut self.targetOfShards {
            std::mem::swap(&mut phase_shard.cmplxDiff, &mut phase_shard.cmplxSame);
        }
        for (_, phase_shard) in &mut self.antiTargetOfShards {
            std::mem::swap(&mut phase_shard.cmplxDiff, &mut phase_shard.cmplxSame);
        }
    }

    fn commute_phase(&mut self, topLeft: Complex, bottomRight: Complex) {
        for (_, phase_shard) in &mut self.targetOfShards {
            if !phase_shard.isInvert {
                return;
            }
            phase_shard.cmplxDiff *= topLeft / bottomRight;
            phase_shard.cmplxSame *= bottomRight / topLeft;
        }
        for (_, phase_shard) in &mut self.antiTargetOfShards {
            if !phase_shard.isInvert {
                return;
            }
            phase_shard.cmplxDiff *= bottomRight / topLeft;
            phase_shard.cmplxSame *= topLeft / bottomRight;
        }
    }

    fn remove_identity_buffers(&mut self, local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>) {
        let mut phase_shard = local_map.iter().next();
        let mut i = 0;
        while let Some((_, buffer)) = phase_shard {
            if !buffer.isInvert && IS_ARG_0(buffer.cmplxDiff) {
                phase_shard = local_map.iter().nth(i);
                i += 1;
                continue;
            }
            phase_shard = local_map.iter().next();
        }
    }

    fn remove_phase_buffers(&mut self, local_map: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>) {
        let mut phase_shard = local_map.iter().next();
        while let Some((partner, _)) = phase_shard {
            self.remove_control(partner.clone());
            phase_shard = local_map.iter().next();
        }
    }

    fn commute_h(&mut self) {
        self.remove_phase_buffers(&mut self.targetOfShards);
        self.remove_phase_buffers(&mut self.antiTargetOfShards);
        self.remove_phase_buffers(&mut self.controlsShards);
        self.remove_phase_buffers(&mut self.antiControlsShards);
    }

    fn is_invert_control(&self) -> bool {
        // TODO: Implement this function
        false
    }

    fn is_invert_target(&self) -> bool {
        // TODO: Implement this function
        false
    }

    fn clear_map_invert_phase(&mut self, shards: &mut HashMap<Rc<QEngineShard>, Rc<PhaseShard>>) {
        for (_, phase_shard) in shards {
            phase_shard.isInvert = false;
        }
    }

    fn clear_invert_phase(&mut self) {
        self.clear_map_invert_phase(&mut self.controlsShards);
        self.clear_map_invert_phase(&mut self.antiControlsShards);
        self.clear_map_invert_phase(&mut self.targetOfShards);
        self.clear_map_invert_phase(&mut self.antiTargetOfShards);
    }

    fn get_qubit_count(&self) -> usize {
        self.unit.as_ref().map_or(1, |u| u.get_qubit_count())
    }

    fn prob(&self) -> f64 {
        if !self.isProbDirty || self.unit.is_none() {
            return (self.amp1.norm_sqr()) as f64;
        }
        self.unit.as_ref().unwrap().prob(self.mapped)
    }

    fn is_clifford(&self) -> bool {
        if let Some(unit) = &self.unit {
            unit.is_clifford(self.mapped)
        } else {
            self.amp0.norm_sqr() <= FP_NORM_EPSILON
                || self.amp1.norm_sqr() <= FP_NORM_EPSILON
                || (self.amp0 - self.amp1).norm_sqr() <= FP_NORM_EPSILON
                || (self.amp0 + self.amp1).norm_sqr() <= FP_NORM_EPSILON
                || (self.amp0 - I_CMPLX * self.amp1).norm_sqr() <= FP_NORM_EPSILON
                || (self.amp0 + I_CMPLX * self.amp1).norm_sqr() <= FP_NORM_EPSILON
        }
    }
}

struct QEngineShardMap {
    shards: Vec<QEngineShard>,
    swapMap: Vec<usize>,
}

impl QEngineShardMap {
    fn new() -> Self {
        QEngineShardMap {
            shards: Vec::new(),
            swapMap: Vec::new(),
        }
    }

    fn new_with_size(size: usize) -> Self {
        let mut shard_map = QEngineShardMap::new();
        for i in 0..size {
            shard_map.swapMap.push(i);
        }
        shard_map
    }

    fn push_back(&mut self, shard: QEngineShard) {
        self.shards.push(shard);
        self.swapMap.push(self.swapMap.len());
    }

    fn insert(&mut self, start: usize, to_insert: &mut QEngineShardMap) {
        let o_size = self.size();
        self.shards.extend(to_insert.shards.drain(..));
        self.swapMap.splice(start..start, to_insert.swapMap.drain(..));
        for lcv in 0..to_insert.size() {
            self.swapMap[start + lcv] += o_size;
        }
    }

    fn erase(&mut self, begin: usize, end: usize) {
        for index in begin..end {
            let offset = self.swapMap[index];
            self.shards.remove(offset);
            for lcv in 0..self.swapMap.len() {
                if self.swapMap[lcv] >= offset {
                    self.swapMap[lcv] -= 1;
                }
            }
        }
        self.swapMap.drain(begin..end);
    }

    fn swap(&mut self, qubit1: usize, qubit2: usize) {
        self.swapMap.swap(qubit1, qubit2);
    }
}

